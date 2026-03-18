import os
import hashlib
import shutil
import subprocess
from datetime import datetime
from PIL import Image, ImageCms, ImageOps

class SocialMediaSimulator:
    def __init__(self, base_output_dir="media_output"):
        self.base_output_dir = base_output_dir

        if not os.path.exists(self.base_output_dir):
            os.makedirs(self.base_output_dir)

    def _ensure_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _get_video_dimensions(self, input_path):
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", input_path
        ]
        try:
            output = subprocess.check_output(cmd).decode("utf-8").strip()
            width, height = map(int, output.split('x'))
            return width, height
        except Exception as e:
            print(f"Error: {e}")
            return None, None

    # =========================================================================
    # FACEBOOK
    # =========================================================================
    def facebook(self, input_path):
        """
        Simulates Facebook's pipeline:
        1. Strips Metadata
        2. Converts to sRGB
        3. Resizes to max 2048px
        4. Applies JPEG compression with 4:2:0 subsampling
        """
        output_dir = os.path.join(self.base_output_dir, "facebook") # No changes needed, single pipeline
        self._ensure_dir(output_dir)

        ext = os.path.splitext(input_path)[1].lower()
        is_image = ext in ['.jpg', '.jpeg', '.png', '.heic', '.webp', '.tiff']
        is_video = ext in ['.mp4', '.mov', '.avi', '.mkv']

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        if is_image:
             output_filename = f"{base_name}.jpg"
        elif is_video:
             output_filename = f"{base_name}.mp4"
        else:
             output_filename = f"{base_name}{ext}"

        output_path = os.path.join(output_dir, output_filename)

        if is_video:
            self._facebook_process_video(input_path, output_path)
        elif is_image:
            self._facebook_process_image(input_path, output_path)
        else:
            print("Unsupported media type for Facebook simulation.")
            return "file type not supported"

    def _facebook_process_video(self, input_path, output_path):
        width, height = self._get_video_dimensions(input_path)
        if width is None or height is None:
            print("Could not retrieve video dimensions.")
            return

        #Downscaling - Cap at 2048px on the longest edge
        scale_filter = "scale='if(gt(a,1),min(2048,iw),-2)':'if(gt(a,1),-2,min(2048,ih))'"
        
        #Chroma Subsampling - Force 4:2:0
        format_filter = "format=yuv420p"
        
        #Loudness normalization - target -16 LUFS as per documentation.
        audio_filter = "loudnorm=I=-16:LRA=11:TP=-1.5"

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", f"{scale_filter},{format_filter}",
            #codec transcoding H.264
            "-c:v", "libx264", 
            "-preset", "fast",
            "-crf", "18", # High quality, variable bitrate approach
            "-maxrate", "4000k", "-bufsize", "8000k",
            #Group Of Pictures (GOP) Standardization - Force keyframe every 2 seconds (assuming 30fps = 60)
            "-g", "60", "-keyint_min", "60", "-sc_threshold", "0",
            "-af", audio_filter,
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
            #EXIF Stripping & Faststart for web
            "-map_metadata", "-1", 
            "-movflags", "+faststart",
            output_path
        ]

        try:
            subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
            print(f"Saved to: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Facebook video pipeline failed: {e}")
    
    def _facebook_process_image(self, input_path, output_path):
        try:
            print(f"\n Facebook Processing: {os.path.basename(input_path)} ")
            original_image = Image.open(input_path)
            
            # 1. Convert to sRGB
            working_image = original_image.convert("RGB")
            try:
                srgb_profile = ImageCms.createProfile("sRGB")
                if "icc_profile" in original_image.info:
                    input_profile = ImageCms.getOpenProfile(original_image.info["icc_profile"])
                    working_image = ImageCms.profileToProfile(working_image, input_profile, srgb_profile, outputMode="RGB")
                else:
                    working_image = ImageCms.profileToProfile(working_image, srgb_profile, srgb_profile, outputMode="RGB")
            except Exception:
                working_image = working_image.convert("RGB")

            # 2. Resize to max 2048px
            max_dimension = 2048
            width, height = working_image.size
            if max(width, height) > max_dimension:
                scale_factor = max_dimension / max(width, height)
                new_size = (int(width * scale_factor), int(height * scale_factor))
                working_image = working_image.resize(new_size, Image.Resampling.LANCZOS)
                print(f"Downscaled to {new_size}")

            working_image.save(output_path, "JPEG", quality=85, optimize=True, subsampling=0)
            print(f"Saved to: {output_path}")

        except Exception as e:
            print(f"Facebook image pipeline failed: {e}")

    # =========================================================================
    # INSTAGRAM
    # =========================================================================
    def instagram(self, input_path, post_type='feed'):
        """
        Simulates Instagram.
        post_type: 'feed', 'reel', or 'story'.
        """
        # Create distinct directories for feed vs story/reel to avoid overwrites
        output_dir_name = "instagram_story_reel" if post_type in ['story', 'reel'] else "instagram_feed"
        output_dir = os.path.join(self.base_output_dir, output_dir_name)
        self._ensure_dir(output_dir)

        ext = os.path.splitext(input_path)[1].lower()
        is_video = ext.lower() in ['.mp4', '.mov', '.avi', '.mkv', '.webm']

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{base_name}.mp4" if is_video else f"{base_name}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        print(f"\n Instagram Processing ({post_type.upper()})")

        if not is_video:
            try:
                with Image.open(input_path) as img:
                    img = img.convert('RGB')
                    width, height = img.size
                    aspect_ratio = width / height
                    target_width = 1080


                    if post_type == 'feed':
                        if aspect_ratio < 0.8:
                            new_height = int(width / 0.8)
                            top = (height - new_height) // 2
                            img = img.crop((0, top, width, top + new_height))
                        elif aspect_ratio > 1.91:
                            new_width = int(height * 1.91)
                            left = (width - new_width) // 2
                            img = img.crop((left, 0, left + new_width, height))
                        
                        new_height = int(target_width / img.size[0] * img.size[1])
                        img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)

                    elif post_type in ['story', 'reel']:
                        target_height = 1920
                        img = ImageOps.fit(img, (target_width, target_height), method=Image.Resampling.LANCZOS)

                    img.save(output_path, "JPEG", quality=80, optimize=True, subsampling=0)
                    print(f"Saved Image: {output_path}")
            except Exception as e:
                print(f"IG pipeline failed: {e}")

        else:
            # Video Logic (FFmpeg)
            vf_filters = []
            audio_channels = "2"
            target_bitrate = "3000k"

            if post_type == 'feed':
                vf_filters = ["scale=1080:-2", "crop=1080:min(ih\\,1350):0:(ih-oh)/2"]
                target_bitrate = "3500k"
            elif post_type in ['story', 'reel']:
                vf_filters = ["scale=1080:-2", "crop=1080:1920:0:(ih-oh)/2"]
                if post_type == 'story': audio_channels = "1"

            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-vf", ",".join(vf_filters),
                "-c:v", "libx264", "-preset", "fast",
                "-b:v", target_bitrate, "-maxrate", target_bitrate, "-bufsize", target_bitrate,
                "-c:a", "aac", "-b:a", "128k", "-ac", audio_channels,
                "-map_metadata", "-1", "-movflags", "+faststart",
                output_path
            ]
            try:
                subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
                print(f"Saved Video: {output_path}")
            except Exception:
                print("IG pipeline failed.")


    # =========================================================================
    # WHATSAPP
    # =========================================================================
    def whatsapp(self, input_path, quality_mode='standard', upload_type='media'):
        """
        Simulates WhatsApp.
        quality_mode: 'standard' or 'high' (HD).
        upload_type: 'media' (Gallery) or 'document' (Files).
        """
        if not os.path.exists(input_path): return

        # Create distinct directories to avoid overwrites based on quality and type
        output_dir_name = f"whatsapp_{quality_mode}" if upload_type == 'media' else "whatsapp_document"
        output_dir = os.path.join(self.base_output_dir, output_dir_name)
        self._ensure_dir(output_dir)

        ext = os.path.splitext(input_path)[1].lower()
        is_image = ext in ['.jpg', '.jpeg', '.png', '.heic', '.webp', '.tiff']
        is_video = ext in ['.mp4', '.mov', '.avi', '.mkv']

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        if upload_type == 'document':
             output_filename = f"{base_name}{ext}"
        elif is_image:
             output_filename = f"{base_name}.jpg"
        elif is_video:
             output_filename = f"{base_name}.mp4"
        else:
             output_filename = f"{base_name}{ext}"

        output_path = os.path.join(output_dir, output_filename)
        print(f"\n WhatsApp Processing ({upload_type.upper()}/{quality_mode.upper()})")

        if upload_type == 'document':
            shutil.copy2(input_path, output_path)
            print(f"Document Copy complete.")
            return

        if is_image:
            self._whatsapp_process_image(input_path, output_path, quality_mode)
        elif is_video:
            self._whatsapp_process_video(input_path, output_path, quality_mode)

    def _whatsapp_process_image(self, input_path, output_path, quality_mode):
        try:
            with Image.open(input_path) as img:
                if img.mode in ('RGBA', 'P'): img = img.convert('RGB')
                
                max_edge = 4096 if quality_mode == 'high' else 1600
                width, height = img.size
                if max(width, height) > max_edge:
                    ratio = max_edge / max(width, height)
                    img = img.resize((int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS)
                
                jpg_quality = 80 if quality_mode == 'high' else 70
                img.save(output_path, 'JPEG', quality=jpg_quality, optimize=True)
                print(f"Saved Image: {output_path}")
        except Exception as e:
            print(f" WA Image failed: {e}")

    def _whatsapp_process_video(self, input_path, output_path, quality_mode):
        scale = "scale=-2:720" if quality_mode == 'high' else "scale=-2:480"
        bitrate = "3M" if quality_mode == 'high' else "1.5M"
        
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx264', '-c:a', 'aac',
            '-vf', f"{scale},fps=30",
            '-b:v', bitrate, '-map_metadata', '-1',
            output_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Saved Video: {output_path}")
        except Exception:
            print(" WA Video failed.")

    # =========================================================================
    # SIGNAL
    # =========================================================================
    def signal(self, input_path, quality_setting='standard', as_document=False):
        """Simulates Signal's privacy-focused pipeline."""
        if not os.path.exists(input_path): return

        # Create distinct directories to avoid overwrites based on settings
        output_dir_name = "signal_document" if as_document else f"signal_{quality_setting}"
        output_dir = os.path.join(self.base_output_dir, output_dir_name)
        self._ensure_dir(output_dir)
        
        ext = os.path.splitext(input_path)[1].lower()
        is_image = ext in ['.jpg', '.jpeg', '.png', '.heic', '.webp', '.tiff']
        is_video = ext in ['.mp4', '.mov', '.avi', '.mkv']

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        if as_document:
             output_filename = f"{base_name}{ext}"
        elif is_image:
             output_filename = f"{base_name}.jpg"
        elif is_video:
             output_filename = f"{base_name}.mp4"
        else:
             output_filename = f"{base_name}{ext}"

        output_path = os.path.join(output_dir, output_filename)

        print(f"\n Signal Processing (Quality: {quality_setting.upper()})")
        
        if as_document:
            shutil.copy2(input_path, output_path)
            print(f"Document Copy complete.")
            return

        if is_image:
            self._signal_process_image(input_path, output_path, quality_setting)
        elif is_video:
            self._signal_process_video(input_path, output_path)
        else:
            print("Unsupported media type for Signal.")

    def _signal_process_image(self, input_path, output_path, quality_setting):
        try:
            with Image.open(input_path) as img:
                if img.mode != 'RGB': img = img.convert('RGB')
                
                max_edge = 4096 if quality_setting == 'high' else 1600
                width, height = img.size
                if max(width, height) > max_edge:
                    ratio = max_edge / max(width, height)
                    img = img.resize((int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS)
                
                # Signal specifically strips metadata in media mode
                img.save(output_path, 'JPEG', quality=80)
                print(f"Metadata stripped. Resized to {img.size}. Saved to: {output_path}")
        except Exception as e:
            print(f"Error: {e}")

    def _signal_process_video(self, input_path, output_path):

        scale = "scale=-2:640" 
        bitrate = "1.5M"       
        
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx264', '-c:a', 'aac',
            '-vf', f"{scale},fps=30",
            '-b:v', bitrate, '-maxrate', bitrate, '-bufsize', f"{float(bitrate[:-1])*2}M",
            '-map_metadata', '-1', 
            output_path
        ]
        print(f"  [Video] Applying aggressive compression (Max 640p, {bitrate} bitrate)...")
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Saved Video: {output_path}")
        except Exception:
            print(" Signal Video failed.")

    # =========================================================================
    # TELEGRAM
    # =========================================================================
    def telegram(self, input_path, as_document=False):
        """Simulates Telegram's cloud-optimized pipeline."""
        if not os.path.exists(input_path): return

        # Create distinct directories to avoid overwrites based on settings
        output_dir_name = "telegram_document" if as_document else "telegram_media"
        output_dir = os.path.join(self.base_output_dir, output_dir_name)
        self._ensure_dir(output_dir)
        
        ext = os.path.splitext(input_path)[1].lower()
        is_image = ext in ['.jpg', '.jpeg', '.png', '.heic', '.webp', '.tiff']
        is_video = ext in ['.mp4', '.mov', '.avi', '.mkv']

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        if as_document:
             output_filename = f"{base_name}{ext}"
        elif is_image:
             output_filename = f"{base_name}.jpg"
        elif is_video:
             output_filename = f"{base_name}.mp4"
        else:
             output_filename = f"{base_name}{ext}"

        output_path = os.path.join(output_dir, output_filename)

        print(f"\n Telegram Processing (Document: {as_document})")

        if as_document:
            shutil.copy2(input_path, output_path)
            print(f"Original quality preserved. Saved to: {output_path}")
            return

        if is_image:
            self._telegram_process_image(input_path, output_path)
        elif is_video:
            self._telegram_process_video(input_path, output_path)


    def _telegram_process_image(self, input_path, output_path):
        try:
            with Image.open(input_path) as img:
                #ensure image is in RGB mode before saving as JPEG
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Telegram aggressive default compression for photos
                max_edge = 1280 
                width, height = img.size
                if max(width, height) > max_edge:
                    ratio = max_edge / max(width, height)
                    img = img.resize((int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS)

                img.save(output_path, 'JPEG', quality=85)
                print(f"Saved Image: {output_path}")
        except Exception as e:
            print(f"Error: {e}")

    def _telegram_process_video(self, input_path, output_path):

        scale = "scale=-2:720" 
        bitrate = "1.8M"  

        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx264', '-c:a', 'aac',
            '-vf', f"{scale},fps=30",
            '-b:v', bitrate, '-maxrate', bitrate, '-bufsize', f"{float(bitrate[:-1])*2}M",
            '-map_metadata', '-1',
            output_path
        ]
        print(f"  [Video] Applying standard compression (Max 720p, {bitrate} bitrate)...")
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Saved Video: {output_path}")
        except Exception:
            print(" Telegram Video failed.")

    # =========================================================================
    # TIKTOK
    # =========================================================================
    def tiktok(self, input_path):
        """Simulates TikTok's aggressive broadcast pipeline."""
        if not os.path.exists(input_path): return

        output_dir = os.path.join(self.base_output_dir, "tiktok") # No changes needed, single pipeline
        self._ensure_dir(output_dir)

        ext = os.path.splitext(input_path)[1].lower()
        is_video = ext in ['.mp4', '.mov', '.avi', '.mkv']

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{base_name}.mp4" if is_video else f"{base_name}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        print(f"\n TikTok Processing")

        if is_video:
            # Scale to 1080x1920, Pad with black, Aggressive bitrate
            video_filters = (
                "scale=1080:1920:force_original_aspect_ratio=decrease,"
                "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,"
                "fps=30"
            )
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-vf', video_filters,
                '-c:v', 'libx264', '-preset', 'veryfast',
                '-b:v', '2500k', '-maxrate', '2500k', '-bufsize', '5000k',
                '-c:a', 'aac', '-b:a', '128k',
                '-pix_fmt', 'yuv420p', '-map_metadata', '-1',
                output_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Transcoded to Vertical. Bitrate changed to 2.5Mbps. Saved to: {output_path}")
        else:
            try:
                with Image.open(input_path) as img:
                    if img.mode != 'RGB': img = img.convert('RGB')
                    
                    target_size = (1080, 1920)
                    img.thumbnail(target_size, Image.Resampling.LANCZOS)
                    
                    background = Image.new('RGB', target_size, (0, 0, 0))
                    offset = ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2)
                    background.paste(img, offset)
                    
                    background.save(output_path, quality=85)
                    
                    print(f"Padded to 9:16 vertical. Saved to: {output_path}")
            except Exception as e:
                print(f"Error: {e}")