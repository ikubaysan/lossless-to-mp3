"""
Recursively convert supported audio files to MP3 320 kbps using FFmpeg.

Features:
- Object-oriented design with type hints.
- Recursively scans multiple input directories.
- Validates all input directories before starting.
- Dry-run mode lists targeted files without modifying anything.
- Preserves metadata and embedded album artwork.
- Deletes the original only after a successful conversion.
- Logs conversion and deletion failures.
- Skips existing MP3 files.
- Avoids overwriting existing MP3 files by default.
- Converts ALL .m4a files, regardless of their internal codec.

Requirements:
    FFmpeg must be installed and available in PATH.

Examples:
    # Convert multiple directories:
    python lossless_to_mp3.py "D:\\Music" "E:\\More Music"

    # List all files that would be converted:
    python lossless_to_mp3.py "D:\\Music" "E:\\More Music" --dry-run

    # Convert to a separate directory:
    python lossless_to_mp3.py "D:\\Music" "E:\\More Music" --output-dir "D:\\MP3"

    # Convert only, without deleting originals:
    python lossless_to_mp3.py "D:\\Music" "E:\\More Music" --keep-originals
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Audio extensions to convert.
#
# M4A files are included regardless of their internal codec.
# This means both ALAC and AAC M4A files will be targeted.
CONVERTIBLE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".flac",
        ".wav",
        ".aiff",
        ".aif",
        ".alac",
        ".ape",
        ".wv",
        ".tta",
        ".tak",
        ".dsf",
        ".dff",
        ".m4a",
    }
)

# Log file written in the current working directory.
LOG_FILE = "lossless_to_mp3.log"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConversionResult:
    """Result of attempting to convert one file."""

    source: Path
    destination: Path
    success: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# File scanner
# ---------------------------------------------------------------------------

class LosslessFileScanner:
    """Finds supported audio files recursively."""

    def __init__(self, root_directory: Path) -> None:
        self.root_directory = root_directory

    def find_files(self) -> list[Path]:
        """
        Return all supported audio files recursively.

        M4A files are included regardless of their internal codec.

        Files are sorted for predictable output.
        """
        files: list[Path] = []

        for path in self.root_directory.rglob("*"):
            if not path.is_file():
                continue

            extension = path.suffix.lower()

            if extension in CONVERTIBLE_EXTENSIONS:
                files.append(path)

        return sorted(files)


# ---------------------------------------------------------------------------
# FFmpeg converter
# ---------------------------------------------------------------------------

class FFmpegConverter:
    """Converts supported audio files to MP3 using FFmpeg."""

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        bitrate: str = "320k",
        overwrite: bool = False,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.bitrate = bitrate
        self.overwrite = overwrite

    def is_available(self) -> bool:
        """Return True if FFmpeg can be found."""
        return shutil.which(self.ffmpeg_path) is not None

    def convert(
        self,
        source: Path,
        destination: Path,
    ) -> ConversionResult:
        """
        Convert one file to MP3.

        - libmp3lame encoder
        - 320 kbps CBR
        - Copy metadata
        - Copy embedded artwork
        - Preserve artwork as an attached picture stream
        """
        destination.parent.mkdir(parents=True, exist_ok=True)

        command: list[str] = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-map",
            "0:v?",
            "-c:a",
            "libmp3lame",
            "-b:a",
            self.bitrate,
            "-map_metadata",
            "0",
            "-id3v2_version",
            "3",
            "-c:v",
            "copy",
            "-disposition:v:0",
            "attached_pic",
        ]

        if self.overwrite:
            command.insert(1, "-y")
        else:
            command.insert(1, "-n")

        command.append(str(destination))

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )

        except OSError as exc:
            return ConversionResult(
                source=source,
                destination=destination,
                success=False,
                error=str(exc),
            )

        if result.returncode != 0:
            error = (
                result.stderr.strip()
                or "FFmpeg returned a non-zero exit code."
            )

            return ConversionResult(
                source=source,
                destination=destination,
                success=False,
                error=error,
            )

        # FFmpeg can return success even if the output is unexpectedly empty.
        if not destination.exists() or destination.stat().st_size == 0:
            return ConversionResult(
                source=source,
                destination=destination,
                success=False,
                error=(
                    "FFmpeg reported success, but the output file "
                    "is missing or empty."
                ),
            )

        return ConversionResult(
            source=source,
            destination=destination,
            success=True,
        )


# ---------------------------------------------------------------------------
# Main conversion manager
# ---------------------------------------------------------------------------

class LosslessToMP3Converter:
    """Coordinates scanning, conversion, and deletion."""

    def __init__(
        self,
        input_directories: list[Path],
        output_directory: Path | None = None,
        ffmpeg_path: str = "ffmpeg",
        bitrate: str = "320k",
        keep_originals: bool = False,
        overwrite: bool = False,
    ) -> None:
        self.input_directories = input_directories
        self.output_directory = output_directory
        self.keep_originals = keep_originals

        self.scanners = [
            LosslessFileScanner(directory)
            for directory in input_directories
        ]

        self.converter = FFmpegConverter(
            ffmpeg_path=ffmpeg_path,
            bitrate=bitrate,
            overwrite=overwrite,
        )

    def get_target_files(self) -> list[tuple[Path, Path]]:
        """
        Return all files that would be targeted.

        Each tuple contains:
            (source_file, input_root_directory)

        Keeping the root directory allows the destination path to be
        calculated correctly when multiple input directories are used.
        """
        files: list[tuple[Path, Path]] = []

        for scanner in self.scanners:
            for source in scanner.find_files():
                files.append((source, scanner.root_directory))

        # If using a separate output directory, don't accidentally scan it.
        if self.output_directory is not None:
            try:
                output_resolved = self.output_directory.resolve()

                files = [
                    (source, root)
                    for source, root in files
                    if output_resolved not in source.resolve().parents
                    and source.resolve() != output_resolved
                ]

            except OSError:
                pass

        return sorted(files, key=lambda item: str(item[0]))

    def get_destination(
        self,
        source: Path,
        input_root: Path,
    ) -> Path:
        """
        Determine the output MP3 path.

        In-place:
            Music/Artist/Album/song.flac
            -> Music/Artist/Album/song.mp3

        Separate directory:
            Music/Artist/Album/song.flac
            -> MP3/Artist/Album/song.mp3
        """
        if self.output_directory is None:
            return source.with_suffix(".mp3")

        relative_path = source.relative_to(input_root)
        destination_relative = relative_path.with_suffix(".mp3")

        return self.output_directory / destination_relative

    def dry_run(self) -> None:
        """List targeted files without modifying anything."""
        files = self.get_target_files()

        print()
        print("Files that would be targeted:")
        print("-" * 80)

        for index, (source, _) in enumerate(files, start=1):
            print(f"{index:>5}. {source}")

        print("-" * 80)
        print(f"Total files: {len(files)}")
        print()
        print("Dry run complete. No files were converted or deleted.")

    def run(self) -> None:
        """Convert all targeted files and delete originals if successful."""
        files = self.get_target_files()

        if not files:
            logging.info("No supported audio files found.")
            return

        logging.info("Found %d audio file(s).", len(files))

        converted_count = 0
        failed_count = 0
        deleted_count = 0
        deletion_failed_count = 0
        skipped_count = 0

        for index, (source, input_root) in enumerate(files, start=1):
            destination = self.get_destination(source, input_root)

            logging.info(
                "[%d/%d] Converting: %s",
                index,
                len(files),
                source,
            )

            # Never overwrite an existing MP3 unless explicitly requested.
            if destination.exists() and not self.converter.overwrite:
                logging.warning(
                    "Skipping because destination already exists: %s",
                    destination,
                )
                skipped_count += 1
                continue

            result = self.converter.convert(source, destination)

            if not result.success:
                failed_count += 1

                logging.error(
                    "FAILED CONVERSION: %s -> %s | %s",
                    source,
                    destination,
                    result.error,
                )
                continue

            converted_count += 1

            logging.info(
                "Conversion successful: %s",
                destination,
            )

            # Only delete after successful conversion.
            if self.keep_originals:
                logging.info(
                    "Keeping original file: %s",
                    source,
                )
                continue

            try:
                source.unlink()

            except OSError as exc:
                deletion_failed_count += 1

                logging.error(
                    "FAILED DELETION: %s | %s",
                    source,
                    exc,
                )
                continue

            deleted_count += 1

            logging.info(
                "Deleted original: %s",
                source,
            )

        print()
        print("Conversion complete.")
        print(f"Converted: {converted_count}")
        print(f"Skipped: {skipped_count}")
        print(f"Failed conversions: {failed_count}")
        print(f"Deleted originals: {deleted_count}")
        print(f"Failed deletions: {deletion_failed_count}")
        print(f"Log file: {LOG_FILE}")
        print()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    """Configure logging to both console and a log file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(
                LOG_FILE,
                encoding="utf-8",
            ),
            logging.StreamHandler(),
        ],
    )


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Recursively convert supported audio files to MP3 320 kbps "
            "using FFmpeg."
        )
    )

    parser.add_argument(
        "input_directories",
        type=Path,
        nargs="+",
        help=(
            "One or more directories to recursively scan. "
            "All directories are validated before conversion begins."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional separate output directory. "
            "If omitted, MP3 files are created beside originals."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "List all targeted files and total count without "
            "converting or deleting anything."
        ),
    )

    parser.add_argument(
        "--keep-originals",
        action="store_true",
        help="Convert files but do not delete the originals.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing MP3 files.",
    )

    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="Path to FFmpeg executable.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Program entry point."""
    configure_logging()

    args = parse_arguments()

    # Validate ALL input directories before doing anything else.
    invalid_directories = [
        directory
        for directory in args.input_directories
        if not directory.is_dir()
    ]

    if invalid_directories:
        logging.error(
            "The following input directories do not exist "
            "or are not directories:"
        )

        for directory in invalid_directories:
            logging.error("  %s", directory)

        logging.error(
            "No conversions were started because one or more "
            "input directories are invalid."
        )

        return 1

    converter = LosslessToMP3Converter(
        input_directories=args.input_directories,
        output_directory=args.output_dir,
        ffmpeg_path=args.ffmpeg,
        keep_originals=args.keep_originals,
        overwrite=args.overwrite,
    )

    if not converter.converter.is_available():
        logging.error(
            "FFmpeg was not found. Install FFmpeg or specify --ffmpeg."
        )
        return 1

    if args.dry_run:
        converter.dry_run()
        return 0

    converter.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())