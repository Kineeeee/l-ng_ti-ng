import os
import subprocess
import yt_dlp
import uuid
import time
import re
import urllib.request

def resolve_redirects(url: str) -> str:
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.geturl()
    except Exception as e:
        print(f"[Warning] Failed to resolve redirects for {url}: {e}")
        return url

def download_douyin_fallback(video_id: str, output_path: str, cookies_file: str = None) -> dict:
    import urllib.parse
    import json
    import http.cookiejar

    # We construct the URL with modal_id to bypass the direct video page captcha
    video_url = f"https://www.douyin.com/jingxuan?modal_id={video_id}"
    print(f"[Module 1] Douyin Fallback: Fetching page HTML from {video_url}...")
    
    cookie_jar = http.cookiejar.MozillaCookieJar()
    if cookies_file and os.path.exists(cookies_file):
        try:
            cookie_jar.load(cookies_file, ignore_discard=True, ignore_expires=True)
            print(f"[Module 1] Douyin Fallback: Loaded cookies from {cookies_file}")
        except Exception as e:
            print(f"[Module 1] Douyin Fallback: Failed to load cookies: {e}")
            
    handler = urllib.request.HTTPCookieProcessor(cookie_jar)
    opener = urllib.request.build_opener(handler)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://www.douyin.com/',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    }
    
    req = urllib.request.Request(video_url, headers=headers)
    with opener.open(req, timeout=15) as response:
        html = response.read().decode('utf-8', errors='ignore')
        
    render_data_match = re.search(r'<script id="RENDER_DATA"[^>]*>(.*?)</script>', html)
    if not render_data_match:
        raise Exception("Could not find RENDER_DATA script tag on the webpage.")
        
    raw_data = render_data_match.group(1)
    decoded_data = urllib.parse.unquote(raw_data)
    data = json.loads(decoded_data)
    
    video_detail = data.get("app", {}).get("videoDetail", {})
    if not video_detail:
        raise Exception("Could not find videoDetail in RENDER_DATA.")
        
    video_block = video_detail.get("video", {})
    if not video_block:
        raise Exception("Could not find video block in videoDetail.")
        
    # Find play URL
    play_url = None
    width = video_block.get("width", 0)
    height = video_block.get("height", 0)
    
    bitrate_list = video_block.get("bitRateList", [])
    if bitrate_list:
        valid_formats = [f for f in bitrate_list if f.get("videoFormat") == "mp4" and f.get("playAddr")]
        if valid_formats:
            valid_formats.sort(key=lambda x: x.get("height", 0), reverse=True)
            best_fmt = None
            for fmt in valid_formats:
                if fmt.get("height", 0) <= 720:
                    best_fmt = fmt
                    break
            if not best_fmt:
                best_fmt = valid_formats[-1] # fallback to smallest if all are larger
                
            play_url = best_fmt["playAddr"][0]["src"]
            width = best_fmt.get("width", width)
            height = best_fmt.get("height", height)
            print(f"[Module 1] Douyin Fallback: Selected format {best_fmt.get('gearName')} ({width}x{height})")
            
    if not play_url and video_block.get("playAddr"):
        play_url = video_block["playAddr"][0]["src"]
        
    if not play_url:
        raise Exception("Could not find any video download URL in RENDER_DATA.")
        
    title = video_detail.get("desc", "douyin_video")
    duration = float(video_block.get("duration", 0)) / 1000.0 # milliseconds to seconds
    
    print(f"[Module 1] Douyin Fallback: Downloading video content from CDN...")
    cdn_req = urllib.request.Request(play_url, headers=headers)
    with opener.open(cdn_req, timeout=30) as response:
        with open(output_path, "wb") as out_file:
            out_file.write(response.read())
            
    print(f"[Module 1] Douyin Fallback: Saved to {output_path}")
    
    return {
        "title": title,
        "duration": duration,
        "width": width,
        "height": height
    }

def check_and_convert_json_cookies():
    """
    Checks for any JSON cookie files in the root workspace and converts them to cookies.txt.
    Supports:
    - cookies.json
    - douyin_cookies.json
    - www.douyin.com_*.json
    """
    import json
    import glob
    import time

    # Files to look for in the current directory (project root)
    possible_files = ["cookies.json", "douyin_cookies.json"]
    possible_files.extend(glob.glob("www.douyin.com_*.json"))

    json_path = None
    newest_time = 0
    for f in possible_files:
        if os.path.exists(f):
            mtime = os.path.getmtime(f)
            if mtime > newest_time:
                newest_time = mtime
                json_path = f

    if not json_path:
        return

    output_path = "cookies.txt"
    # Check if cookies.txt is already newer than the JSON file to avoid redundant conversion
    if os.path.exists(output_path) and os.path.getmtime(output_path) >= newest_time:
        return

    print(f"[Module 1] Found JSON cookie file: {json_path}. Converting to Netscape format...")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "cookies" in data:
            cookies = data["cookies"]
        elif isinstance(data, list):
            cookies = data
        else:
            print(f"[Warning] Invalid cookie format in {json_path}")
            return

        lines = [
            "# Netscape HTTP Cookie File",
            "# This file is generated. Do not edit.",
            ""
        ]

        for cookie in cookies:
            domain = cookie.get("domain", "")
            if not domain:
                continue

            if "douyin.com" in domain:
                domain = ".douyin.com"
                flag = "TRUE"
            else:
                host_only = cookie.get("hostOnly", False)
                flag = "FALSE" if host_only else "TRUE"

            path = cookie.get("path", "/")
            secure = "TRUE" if cookie.get("secure", False) else "FALSE"

            expiration_date = cookie.get("expirationDate")
            if expiration_date is None:
                expiration = int(time.time()) + 31536000
            else:
                expiration = int(expiration_date)

            name = cookie.get("name", "")
            value = cookie.get("value", "")

            if not name and not value:
                continue

            line = f"{domain}\t{flag}\t{path}\t{secure}\t{expiration}\t{name}\t{value}"
            lines.append(line)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"[Module 1] Successfully converted {len(cookies)} cookies to {output_path}")
    except Exception as e:
        print(f"[Warning] Failed to convert JSON cookies: {e}")

def download_video(video_url: str, output_dir: str = "output") -> dict:
    """
    Downloads the video using yt-dlp and extracts a 16kHz mono audio WAV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    check_and_convert_json_cookies()
    
    # Normalize Douyin URL
    if "douyin.com" in video_url:
        video_url = resolve_redirects(video_url)
        modal_match = re.search(r'modal_id=(\d+)', video_url)
        if modal_match:
            video_id = modal_match.group(1)
            video_url = f"https://www.douyin.com/video/{video_id}"
            print(f"[Module 1] Normalized Douyin URL to: {video_url}")

    # Generate unique ID for this download session
    job_id = str(uuid.uuid4())[:8]
    base_path = os.path.join(output_dir, job_id)
    video_path = f"{base_path}.mp4"
    audio_path = f"{base_path}.wav"
    
    # yt-dlp options
    ydl_opts = {
        'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]',
        'outtmpl': video_path,
        'noplaylist': True,
        'quiet': False
    }
    
    # Try reading cookies from file if configured, or default cookies.txt in workspace
    cookies_file = os.environ.get("YT_COOKIES_FILE", "cookies.txt").strip()
    browser_for_cookies = os.environ.get("YT_COOKIES_BROWSER", "").strip()

    if browser_for_cookies:
        ydl_opts['cookiesfrombrowser'] = (browser_for_cookies,)
        print(f"[Module 1] Using cookies from browser: {browser_for_cookies}")
        if cookies_file:
            ydl_opts['cookiefile'] = cookies_file
            print(f"[Module 1] Will save/update browser cookies to file: {cookies_file}")
    elif cookies_file and os.path.exists(cookies_file):
        ydl_opts['cookiefile'] = cookies_file
        print(f"[Module 1] Using cookies from file: {cookies_file}")

    t0 = time.time()
    try:
        print(f"[Module 1] Downloading video from {video_url}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            duration = info_dict.get('duration', 0)
            width = info_dict.get('width', 0)
            height = info_dict.get('height', 0)
            resolution = (width, height)
            
            # If the file extension downloaded is not mp4, it's possible it might differ
            # But our format string heavily enforces mp4. We'll use the returned filename just in case.
            actual_video_path = ydl.prepare_filename(info_dict)
            if not os.path.exists(actual_video_path):
                actual_video_path = video_path # Fallback

        t_download = time.time() - t0
        print(f"[Module 1] Video downloaded in {t_download:.1f}s")
    except Exception as e:
        # Check if Douyin URL and try fallback
        if "douyin.com" in video_url:
            print(f"[Warning] yt-dlp failed to download Douyin video: {e}")
            print(f"[Module 1] Invoking custom Douyin fallback downloader...")
            try:
                # Extract video ID from normalized URL
                video_id_match = re.search(r'video/(\d+)', video_url)
                if not video_id_match:
                    video_id_match = re.search(r'modal_id=(\d+)', video_url)
                
                if video_id_match:
                    video_id = video_id_match.group(1)
                    fallback_res = download_douyin_fallback(video_id, video_path, cookies_file)
                    t_download = time.time() - t0
                    print(f"[Module 1] Video downloaded via fallback in {t_download:.1f}s")
                    actual_video_path = video_path
                    duration = fallback_res["duration"]
                    resolution = (fallback_res["width"], fallback_res["height"])
                else:
                    raise Exception("Could not extract Douyin video ID from URL")
            except Exception as fe:
                print(f"[Error] Douyin fallback downloader failed: {fe}")
                raise e
        else:
            raise e

    t_download = time.time() - t0
    print(f"[Module 1] Video downloaded in {t_download:.1f}s")

    print(f"[Module 1] Extracting 16kHz mono audio to {audio_path}...")
    t1 = time.time()
    # Extract audio using ffmpeg with multithreading
    # -threads 0: auto-detect optimal thread count
    command = [
        "ffmpeg", "-y",
        "-threads", "0",
        "-i", actual_video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ]
    
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t_extract = time.time() - t1
    
    print(f"[Module 1] Audio extracted in {t_extract:.1f}s")
    print(f"[Module 1] Download complete. Duration: {duration}s, Resolution: {resolution}")
    print(f"[Module 1] Total time: {t_download + t_extract:.1f}s")

    return {
        "video_path": actual_video_path,
        "audio_path": audio_path,
        "duration_sec": float(duration),
        "resolution": resolution
    }

if __name__ == "__main__":
    # Test block
    import sys
    if len(sys.argv) > 1:
        res = download_video(sys.argv[1])
        print("Result:", res)
