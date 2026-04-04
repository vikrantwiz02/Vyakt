document.addEventListener('DOMContentLoaded', () => {
    const startCameraButton = document.getElementById('startCameraButton');
    const closeCameraButton = document.getElementById('closeCameraButton');
    const cameraFeedContainer = document.getElementById('cameraFeedContainer');
    const cameraVideo = document.getElementById('cameraVideo');
    const cameraCanvas = document.getElementById('cameraCanvas');
    const annotatedFeed = document.getElementById('annotatedFeed');

    const predictedText = document.getElementById('predictedText');
    const predictionConfidence = document.getElementById('predictionConfidence');
    const finalText = document.getElementById('finalText');

    const startCapture = document.getElementById('startCapture');
    const stopCapture = document.getElementById('stopCapture');

    let currentAudio = null;
    let lastSpoken = "";
    let stopInProgress = false;
    let mediaStream = null;
    let sending = false;
    let frameTimer = null;

    const MAX_UPLOAD_WIDTH = 512;
    const MAX_UPLOAD_HEIGHT = 384;
    const MIN_FRAME_DELAY_MS = 100;
    const MAX_FRAME_DELAY_MS = 300;
    const DEFAULT_FRAME_DELAY_MS = 140;

    function scheduleNextFrame(delayMs = DEFAULT_FRAME_DELAY_MS) {
        if (frameTimer) {
            clearTimeout(frameTimer);
        }
        frameTimer = setTimeout(sendFrame, Math.max(0, delayMs));
    }

    function hideLivePrediction() {
        predictedText.innerText = "";
        predictionConfidence.innerText = "";
    }

    function speakText(text) {
        fetch('/text_to_speech', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        })
            .then((res) => res.json())
            .then((data) => {
                if (data.audio_url) {
                    if (currentAudio) {
                        currentAudio.pause();
                        currentAudio.currentTime = 0;
                    }
                    currentAudio = new Audio(data.audio_url);
                    currentAudio.play();
                }
            })
            .catch(() => {});
    }

    async function sendFrame() {
        if (sending || !cameraVideo.videoWidth) return;
        sending = true;
        const startedAt = performance.now();

        try {
            const scale = Math.min(
                MAX_UPLOAD_WIDTH / cameraVideo.videoWidth,
                MAX_UPLOAD_HEIGHT / cameraVideo.videoHeight,
                1
            );
            const uploadWidth = Math.max(1, Math.round(cameraVideo.videoWidth * scale));
            const uploadHeight = Math.max(1, Math.round(cameraVideo.videoHeight * scale));

            if (cameraCanvas.width !== uploadWidth || cameraCanvas.height !== uploadHeight) {
                cameraCanvas.width = uploadWidth;
                cameraCanvas.height = uploadHeight;
            }

            const ctx = cameraCanvas.getContext('2d');
            ctx.drawImage(cameraVideo, 0, 0, uploadWidth, uploadHeight);
            const dataUrl = cameraCanvas.toDataURL('image/jpeg', 0.7);

            const res = await fetch('/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataUrl }),
            });
            const data = await res.json();

            // Show annotated frame with landmarks
            if (data.annotated_frame) {
                annotatedFeed.src = data.annotated_frame;
                annotatedFeed.style.display = 'block';
            }

            const text = (data.text || '').trim();
            const confidence = Number(data.confidence || 0);
            const state = data.state || 'idle';

            if (!text) {
                hideLivePrediction();
            } else {
                predictedText.innerText = `Live: ${text}`;
                predictionConfidence.innerText = `Confidence: ${Math.round(confidence * 100)}%`;

                if (state === 'predicted' && text !== lastSpoken) {
                    speakText(text);
                    lastSpoken = text;
                }
            }
        } catch {
            hideLivePrediction();
        } finally {
            sending = false;
            const elapsed = performance.now() - startedAt;
            const nextDelay = Math.min(
                MAX_FRAME_DELAY_MS,
                Math.max(MIN_FRAME_DELAY_MS, Math.round(elapsed * 1.15))
            );
            if (mediaStream) {
                scheduleNextFrame(nextDelay);
            }
        }
    }

    startCameraButton.addEventListener('click', async () => {
        try {
            await fetch('/reset_capture_state', { method: 'POST' });
            await fetch('/reset_frame_state', { method: 'POST' });
        } catch {}

        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: 640, max: 960 },
                    height: { ideal: 480, max: 720 },
                    frameRate: { ideal: 24, max: 30 },
                },
                audio: false,
            });
        } catch (err) {
            alert('Camera permission denied or not available: ' + err.message);
            return;
        }

        cameraVideo.srcObject = mediaStream;
        cameraFeedContainer.style.display = 'block';
        finalText.innerText = '';

        startCameraButton.style.display = 'none';
        closeCameraButton.style.display = 'inline-flex';
        startCapture.style.display = 'inline-flex';
        stopCapture.style.display = 'inline-flex';

        // Wait for video to actually start playing before sending frames
        cameraVideo.onloadeddata = () => {
            sendFrame();
        };
    });

    closeCameraButton.addEventListener('click', () => {
        if (frameTimer) {
            clearTimeout(frameTimer);
            frameTimer = null;
        }

        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        cameraVideo.srcObject = null;
        annotatedFeed.src = '';
        annotatedFeed.style.display = 'none';
        cameraFeedContainer.style.display = 'none';

        startCameraButton.style.display = 'inline-flex';
        closeCameraButton.style.display = 'none';
        startCapture.style.display = 'none';
        stopCapture.style.display = 'none';

        if (currentAudio) {
            currentAudio.pause();
            currentAudio.currentTime = 0;
        }

        hideLivePrediction();
        finalText.innerText = '';
        lastSpoken = '';
    });

    startCapture.addEventListener('click', async () => {
        await fetch('/start_capture', { method: 'POST' });

        finalText.innerText = 'Capturing...';
        startCapture.classList.add('capturing');

        startCapture.disabled = true;
        stopCapture.disabled = false;
    });

    stopCapture.addEventListener('click', async () => {
        if (stopInProgress) {
            return;
        }
        stopInProgress = true;
        stopCapture.disabled = true;

        const res = await fetch('/stop_capture', { method: 'POST' });
        const data = await res.json();

        if (data.words && data.words.trim() !== '') {
            finalText.innerText = `Final: ${data.words}`;
            speakText(data.words);
        } else {
            finalText.innerText = '';
        }

        startCapture.classList.remove('capturing');

        startCapture.disabled = false;
        stopCapture.disabled = true;
        stopInProgress = false;
    });
});
