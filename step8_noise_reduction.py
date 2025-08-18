import os
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
import argparse

class NoiseReducer:
    """
    Step 8: Background noise reduction for voice samples.

    Modes:
      - quick: fast denoise using afftdn + highpass/lowpass
      - profile: apply a noise profile (afir) you provide via --noise-file
    """

    def __init__(
        self,
        output_dir: str = "voice_analysis",
        mode: str = "quick",
        noise_profile_file: Optional[str] = None,
        sample_rate: int = 16000,
        highpass_hz: int = 80,
        lowpass_hz: int = 6000,
        afftdn_nr: float = 24.0,
        afftdn_nf: float = 12.0,
        timeout: int = 120
    ):
        self.output_dir = output_dir
        self.mode = mode.lower().strip()
        self.noise_profile_file = noise_profile_file
        self.sample_rate = sample_rate
        self.highpass_hz = highpass_hz
        self.lowpass_hz = lowpass_hz
        self.afftdn_nr = afftdn_nr  # Noise reduction dB
        self.afftdn_nf = afftdn_nf  # Noise floor dB
        self.timeout = timeout

        self.denoised_dir = os.path.join(output_dir, "denoised_audio")
        os.makedirs(self.denoised_dir, exist_ok=True)

        if self.mode not in {"quick", "profile"}:
            raise ValueError("mode must be 'quick' or 'profile'")

        if self.mode == "profile" and not self.noise_profile_file:
            raise ValueError("profile mode requires noise_profile_file")

        print("🎛️ NoiseReducer initialized")
        print(f"📁 Output denoised dir: {self.denoised_dir}")
        print(f"⚙️ Mode: {self.mode}")

    def process_directory(self, input_dir: str) -> List[Dict]:
        """
        Process all audio files in input_dir and write denoised WAV files.
        Returns a list of result dicts per file.
        """
        if not os.path.exists(input_dir):
            print(f"❌ Input directory not found: {input_dir}")
            return []

        audio_files: List[Path] = []
        # WAV-only to match your requirement
        for ext in (".wav",):
            audio_files.extend(Path(input_dir).glob(f"*{ext}"))

        if not audio_files:
            print(f"❌ No WAV files found in: {input_dir}")
            return []

        print(f"🎧 Found {len(audio_files)} WAV files for noise reduction")

        results: List[Dict] = []
        for i, audio_path in enumerate(sorted(audio_files), 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {audio_path.name}")
            out_path = self._build_output_path(audio_path)
            success, status = self._denoise_file(str(audio_path), out_path)
            results.append({
                "input_file": str(audio_path),
                "output_file": out_path if success else "",
                "status": status,
                "mode": self.mode
            })

            if success:
                print(f"✅ Saved: {os.path.basename(out_path)}")
            else:
                print(f"❌ Failed: {status}")

            time.sleep(0.2)

        print("\n🎛️ Noise reduction completed")
        print(f"✅ Success: {sum(1 for r in results if r['output_file'])}")
        print(f"❌ Failed: {sum(1 for r in results if not r['output_file'])}")
        print(f"📁 Denoised files: {self.denoised_dir}")
        return results

    def _build_output_path(self, audio_path: Path) -> str:
        base = audio_path.stem
        out_name = f"{base}_denoised.wav"
        return os.path.join(self.denoised_dir, out_name)

    def _denoise_file(self, input_file: str, output_file: str) -> Tuple[bool, str]:
        """
        Run ffmpeg with selected filter chain.
        Always outputs mono 16kHz WAV.
        """
        # Basic input checks to avoid ffmpeg on invalid files
        if not os.path.exists(input_file) or os.path.getsize(input_file) < 1024:
            return False, "input_missing_or_too_small"

        if self.mode == "quick":
            af = (
                f"highpass=f={self.highpass_hz},"
                f"lowpass=f={self.lowpass_hz},"
                f"afftdn=nr={self.afftdn_nr}:nf={self.afftdn_nf}:nt=w:om=o,"
                f"dynaudnorm=f=75:g=15:p=0.9:m=10"
            )
        else:
            if not os.path.exists(self.noise_profile_file):
                return False, "noise_profile_not_found"
            af = (
                f"highpass=f={self.highpass_hz},"
                f"lowpass=f={self.lowpass_hz},"
                f"afir=ir='{self.noise_profile_file}':dry=1:wet=0,"
                f"afftdn=nr={max(self.afftdn_nr - 6, 0)}:nf={self.afftdn_nf}:nt=w:om=o,"
                f"dynaudnorm=f=75:g=15:p=0.9:m=10"
            )

        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_file,
            "-ac", "1",
            "-ar", str(self.sample_rate),
            "-af", af,
            "-c:a", "pcm_s16le",
            output_file
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            if result.returncode != 0:
                return False, f"ffmpeg_error: {result.stderr[:200]}"
            if not os.path.exists(output_file) or os.path.getsize(output_file) < 8000:
                return False, "output_file_invalid_or_too_small"
            return True, "ok"
        except subprocess.TimeoutExpired:
            return False, "ffmpeg_timeout"
        except FileNotFoundError:
            return False, "ffmpeg_not_installed"
        except Exception as e:
            return False, f"exception: {str(e)[:120]}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 8 - Background Noise Reduction")
    parser.add_argument("--input-dir", required=True, help="Directory with WAV files to denoise")
    parser.add_argument("--output-dir", default="voice_analysis", help="Output base dir (denoised files in subfolder)")
    parser.add_argument("--mode", choices=["quick", "profile"], default="quick", help="Denoise mode")
    parser.add_argument("--noise-file", help="Noise profile file (WAV) for 'profile' mode")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--highpass", type=int, default=80, help="Highpass cutoff Hz")
    parser.add_argument("--lowpass", type=int, default=6000, help="Lowpass cutoff Hz")
    parser.add_argument("--nr", type=float, default=24.0, help="afftdn noise reduction dB")
    parser.add_argument("--nf", type=float, default=12.0, help="afftdn noise floor dB")
    args = parser.parse_args()

    reducer = NoiseReducer(
        output_dir=args.output_dir,
        mode=args.mode,
        noise_profile_file=args.noise_file,
        sample_rate=args.sr,
        highpass_hz=args.highpass,
        lowpass_hz=args.lowpass,
        afftdn_nr=args.nr,
        afftdn_nf=args.nf
    )

    reducer.process_directory(args.input_dir)
