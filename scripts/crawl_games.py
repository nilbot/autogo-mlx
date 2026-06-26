#!/usr/bin/env python3
"""Ranked Human Go Games Crawler from OGS.

Queries the Online-Go.com (OGS) public REST API to retrieve game records (SGF)
matching target board sizes and Elo/rating brackets, utilizing active bot accounts
and human player game histories.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import httpx


# Active bot usernames on OGS mapped to target brackets
BRACKET_USERS = {
    500: ["amybot", "doge_bot_3"],
    1500: ["Kugutsu", "GnuGo", "doge_bot_1"],
    2200: ["Kugutsu", "pachi", "doge_bot_4"],
    2800: ["kata-bot", "Kugutsu"],
}


def get_bracket_rating_range(bracket: int) -> tuple[float, float]:
    """Returns the rating bounds for a given Elo target bracket.

    Args:
        bracket: Target Elo bracket (500, 1500, 2200, 2800).

    Returns:
        A tuple (min_rating, max_rating) representing rating bounds.
    """
    if bracket == 500:
        return 300.0, 800.0
    elif bracket == 1500:
        return 1300.0, 1700.0
    elif bracket == 2200:
        return 2000.0, 2400.0
    elif bracket == 2800:
        return 2500.0, 5000.0
    else:
        raise ValueError(f"Unknown Elo bracket: {bracket}")


def resolve_player_id(client: httpx.Client, username: str) -> int | None:
    """Queries the OGS players API to resolve a username to a player ID.

    Args:
        client: HTTPX client.
        username: The OGS username.

    Returns:
        The numerical player ID, or None if not found.
    """
    url = "https://online-go.com/api/v1/players"
    params = {"username": username}

    for attempt in range(3):
        try:
            response = client.get(url, params=params)
            if response.status_code == 429:
                time.sleep((attempt + 1) * 2.0)
                continue
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            for r in results:
                if r.get("username", "").lower() == username.lower():
                    return int(r["id"])
            break
        except httpx.HTTPError as e:
            # We fail gracefully here and retry, returning None if all attempts fail.
            # If all target usernames fail to resolve, the main crawl loop will catch
            # the empty player_ids list and report a hard failure.
            print(
                f"Warning: HTTP error resolving player '{username}' on attempt {attempt + 1}: {e}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep((attempt + 1) * 1.0)
            
    return None


def crawl_games(
    board_size: int,
    bracket: int,
    num_games: int,
    save_dir: Path,
) -> None:
    """Crawls Go games from OGS matching the target criteria and saves them.

    Args:
        board_size: Go board side length (9 or 19).
        bracket: Target Elo bracket (500, 1500, 2200, 2800).
        num_games: Maximum number of games to download.
        save_dir: Directory where the SGF files should be saved.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    min_rating, max_rating = get_bracket_rating_range(bracket)

    print(
        f"Starting crawl for board_size={board_size}x{board_size}, bracket={bracket} Elo "
        f"(rating range: {min_rating}-{max_rating}), target: {num_games} games.",
        flush=True,
    )

    headers = {
        "User-Agent": "AutoGo-MLX-SFT-Crawler/0.1 (Nils; ersi.ni@nilbot.net)"
    }
    client = httpx.Client(headers=headers, timeout=15.0)

    # 1. Resolve player IDs for target bots/players
    target_usernames = BRACKET_USERS.get(bracket, [])
    player_ids = []
    print(f"Resolving player IDs for usernames: {target_usernames}...", flush=True)
    for username in target_usernames:
        p_id = resolve_player_id(client, username)
        if p_id:
            print(f"-> Resolved username {username} to ID: {p_id}", flush=True)
            player_ids.append(p_id)
        time.sleep(0.5)

    if not player_ids:
        print("ERROR: Could not resolve any player IDs for target usernames.", file=sys.stderr)
        client.close()
        sys.exit(1)

    downloaded = 0
    processed_game_ids = set()

    try:
        # 2. Query game history for resolved players
        for p_id in player_ids:
            if downloaded >= num_games:
                break

            games_url = f"https://online-go.com/api/v1/players/{p_id}/games"
            params: dict[str, str | int] = {}
            visited_urls = set()

            while games_url and downloaded < num_games:
                if games_url in visited_urls:
                    print(f"Warning: URL {games_url} already visited for player {p_id}. Breaking loop.", flush=True)
                    break
                visited_urls.add(games_url)
                print(f"Fetching games history for player {p_id} from {games_url}...", flush=True)
                
                for attempt in range(5):
                    try:
                        response = client.get(games_url, params=params)
                        if response.status_code == 429:
                            time.sleep((attempt + 1) * 2.0)
                            continue
                        response.raise_for_status()
                        break
                    except httpx.HTTPError as e:
                        if attempt == 4:
                            print(f"Failed to fetch games for player {p_id}: {e}", file=sys.stderr, flush=True)
                            break
                        time.sleep((attempt + 1) * 1.5)
                else:
                    break

                params = {}  # Clear params for paginated URLs
                games_data = response.json()
                results = games_data.get("results", [])
                if not results:
                    break

                for g in results:
                    if downloaded >= num_games:
                        break

                    game_id = g.get("id")
                    if not game_id:
                        continue

                    if game_id in processed_game_ids:
                        continue
                    processed_game_ids.add(game_id)

                    # Only crawl completed games
                    if not g.get("ended"):
                        continue

                    width = g.get("width")
                    height = g.get("height")
                    if width != board_size or height != board_size:
                        continue

                    # Filter ratings from nested structure
                    players = g.get("players", {})
                    black_rating = float(players.get("black", {}).get("ratings", {}).get("overall", {}).get("rating", 0.0))
                    white_rating = float(players.get("white", {}).get("ratings", {}).get("overall", {}).get("rating", 0.0))
                    avg_rating = (black_rating + white_rating) / 2.0

                    # Allow a wider range for rating fluctuations
                    if not (min_rating - 400 <= avg_rating <= max_rating + 400):
                        continue

                    sgf_file = save_dir / f"ogs_{game_id}.sgf"
                    if sgf_file.exists():
                        downloaded += 1
                        continue

                    # Download SGF file
                    sgf_url = f"https://online-go.com/api/v1/games/{game_id}/sgf"
                    print(f"Downloading SGF for Game {game_id} (B: {black_rating:.0f}, W: {white_rating:.0f})...", flush=True)

                    for attempt in range(5):
                        try:
                            sgf_resp = client.get(sgf_url)
                            if sgf_resp.status_code == 429:
                                time.sleep((attempt + 1) * 2.0)
                                continue
                            sgf_resp.raise_for_status()
                            break
                        except httpx.HTTPError as e:
                            if attempt == 4:
                                print(f"Failed to download SGF for game {game_id}: {e}", file=sys.stderr, flush=True)
                                break
                            time.sleep((attempt + 1) * 1.0)
                    else:
                        continue

                    sgf_file.write_text(sgf_resp.text, encoding="utf-8")
                    downloaded += 1
                    time.sleep(0.5)

                games_url = games_data.get("next")

    finally:
        client.close()

    if downloaded == 0:
        print("ERROR: Crawl failed to download any valid SGF games.", file=sys.stderr)
        sys.exit(1)

    print(f"Crawl completed! Downloaded {downloaded} SGF files to {save_dir}.", flush=True)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="OGS Go Games Crawler")
    parser.add_argument(
        "--board-size", type=int, default=9, choices=[9, 19], help="Board size"
    )
    parser.add_argument(
        "--bracket", type=int, required=True, choices=[500, 1500, 2200, 2800], help="Elo bracket"
    )
    parser.add_argument(
        "--num-games", type=int, default=10, help="Number of games to download"
    )
    parser.add_argument(
        "--save-dir", type=str, required=True, help="Directory to save SGF files"
    )
    args = parser.parse_args()

    crawl_games(
        board_size=args.board_size,
        bracket=args.bracket,
        num_games=args.num_games,
        save_dir=Path(args.save_dir),
    )


if __name__ == "__main__":
    main()
