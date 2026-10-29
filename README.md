# TakeSync

Turns several OBS recordings of the same song section into one automatically cut,
vertical **1080x1920** performance video. It lines the takes up using the FL Studio
playback audio that OBS recorded, cuts between them every 1–3 seconds, crops 16:9
to 9:16, and keeps one continuous reference soundtrack. That soundtrack is there so
you can line up TikTok's "Add Sound" by ear.

Local only: no accounts, no internet (apart from the one-time setup), and your
recordings are never modified.

---

## Install on Windows (one time)

1. **Install Python.** Open **PowerShell** (Start menu → type `PowerShell`) and run:

   ```bash
   winget install -e --id Python.Python.3.12
   ```

2. **Install FFmpeg** in the same PowerShell window:

   ```bash
   winget install -e --id Gyan.FFmpeg
   ```

3. **Close PowerShell** so Windows picks up the new programs.

4. **Get TakeSync.** On the GitHub page for this repo, click **Code → Download ZIP**,
   then extract it somewhere simple, for example `C:\TakeSync`.

5. **Double-click `TakeSync.bat`.** The first start takes a minute or two while it
   sets up. After that it opens straight away.

If FFmpeg can't be found, you can also put `ffmpeg.exe` and `ffprobe.exe` into a folder
named `ffmpeg\bin` inside the TakeSync folder.

## Using it

1. **Recordings.** Drag your OBS files into the table, or click **Add files...**.
2. **Reference.** TakeSync picks the longest take as the song timeline (the
   "reference") and marks it *(auto-picked)*. You can switch to another take. If you
   have the song section exported from FL Studio as WAV/MP3, click **Use audio
   file...** (or drag it into the window). That gives the most reliable sync and the
   cleanest soundtrack.
3. Click **Analyze**. Each take gets a status:
   | Status | Meaning |
   |---|---|
   | `REFERENCE` | This take defines the timeline |
   | `OK` | Offset found and confirmed in many places across the take |
   | `LOW` | Probably right, but the match is weak. **Not used** until you tick *Use* |
   | `AMBIGUOUS` | Several positions fit, usually because a section repeats. **Not used** until you tick *Use* or type the right offset |
   | `FAILED` | The audio doesn't match (wrong song, silent track). Type an offset to use it anyway |
   | `MANUAL` | You typed the offset yourself |

   You can type any offset in the **Offset (ms)** column, for example `-2500.25`.
   **Reset offset** puts back what Analyze found.
4. **Excerpt.** Start and End are filled in with the part of the song that *every*
   used take covers. Change them if you want, using a format like `0:12.500`.
5. **Generate Preview** plans the cuts and renders a quick low-res draft. Pacing
   **Natural** cuts about every 2.6 s and **Fast** about every 1.3 s. Cuts land on
   detected beats when the beat detection is confident. Changing the **Seed** gives a
   different (but repeatable) arrangement.
6. **Edit the cut list** if you want. You can type a new start time for any cut,
   pick a different take from the dropdown, or use **Split** / **Merge**. Click a row
   to jump the preview there. **Update Preview** renders your edited list.
7. **Export...** writes the final 1080x1920 MP4 (H.264 CRF 17, AAC 320k). TakeSync
   checks the file automatically and shows the result in the log.

**Crop centre** moves the 9:16 window left or right if you weren't centred.
**Camera delay**: only change this if your *raw OBS recordings* already look out of
sync (lips late). It works like OBS's "Sync Offset".

---

## How the timing works (one convention everywhere)

* **Media time**: a file's own clock, counted from its first timestamp. This is what
  FFmpeg seeks to.
* **Song time**: the reference's clock. With an audio file, it is the time since
  that file's first sample. With a reference take, it is that take's media time.
* **Offset** (per take): `song time = media time + offset`.
  * A take that started recording 2.5 s *before* the song position you began
    playback from has offset `-2500 ms`.
  * A take that starts 7 s into the song has `+7000 ms`.
* Output frame `j` shows song time `start + j / fps`. For each take, TakeSync picks
  the source frame nearest to `song time - offset + camera delay`. Nothing is ever
  sped up or slowed down.
* The soundtrack is cut from the reference in **one piece**, starting at the
  excerpt start. It is never stitched at cut points.

## How sync is found

1. Mono 16 kHz copies of each audio track, high-passed at 300 Hz (kicks and sub-bass
   repeat in every song, so they don't help tell positions apart). These copies are
   only used for analysis, never for the output.
2. Coarse search over the whole signal plus several 4-second pieces of the take,
   which gives a list of candidate offsets.
3. Every candidate is checked in up to 24 windows spread across the whole overlap.
   A window only counts if its own best match lands within 1 ms of the candidate.
   The best candidate wins. If a different offset explains the audio almost as well,
   the take is flagged **AMBIGUOUS** instead of being guessed.
4. The final offset is the median of the agreeing windows, with sub-sample
   precision. Clock drift across the take is measured and reported.
5. 1-second windows mark where the take's audio really follows the song. Pre-roll
   before you pressed play, stopped playback and mic-only parts are excluded from
   the cuts.

---

## Command line (for testing and checking real footage)

Open a terminal in the TakeSync folder and use `.venv\Scripts\python.exe`:

```bash
.venv\Scripts\python.exe -m takesync inspect "C:\Recordings\take1.mp4" "C:\Recordings\take2.mp4"
```

```bash
.venv\Scripts\python.exe -m takesync analyze "C:\Recordings\take1.mp4" "C:\Recordings\take2.mp4" "C:\Recordings\take3.mp4"
```

```bash
.venv\Scripts\python.exe -m takesync diagnose "C:\Recordings\take1.mp4" "C:\Recordings\take2.mp4" "C:\Recordings\take3.mp4" --out "C:\Recordings\takesync_check"
```

```bash
.venv\Scripts\python.exe -m takesync export "C:\Recordings\take1.mp4" "C:\Recordings\take2.mp4" "C:\Recordings\take3.mp4" --out "C:\Recordings\final.mp4" --pacing fast
```

`diagnose` writes three files:

* `alignment_report.txt`: offset, confidence and usable range per take.
* `contact_sheet.png`: one row per moment in the song, one column per take (in the
  order you listed them). **Your mouth should be in the same position across each
  row.**
* `diagnostic.mp4`: switches take every 2 s, with the take name burned in, over
  the continuous soundtrack. **Watch for lip-sync and for any jump at the switches.**

Useful export options: `--ref song.wav`, `--ref-take 2`, `--offset 3=-2500.5`
(take 3, in ms), `--accept 3` (use a flagged take), `--exclude 2`,
`--start 0:12 --end 0:45`, `--crop-center 0.45`, `--camera-delay-ms 80`,
`--preview`, `--plan-out cuts.json` / `--plan-in cuts.json`, `--overwrite`.

## Tests

```bash
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

```bash
.venv\Scripts\python.exe -m pytest tests -q
```

The tests build synthetic "OBS recordings" with known offsets. Every video frame
carries a barcode that encodes which take it is and the exact song time being
performed. Exports are then decoded frame by frame to prove that each output frame
shows the planned take at the planned song time.

## Limitations (v1)

* One continuous performance per file. Several takes inside one long recording is
  not supported yet (planned as an optional Phase 5).
* No HDR tone-mapping (10-bit/HDR input gets a warning).
* If two takes cover different parts of the song, the excerpt is limited to what
  the reference covers, because the soundtrack comes from the reference in one piece.
* Subjective cut quality and real-footage lip-sync have to be checked by eye. Use
  `diagnose` or the preview.
