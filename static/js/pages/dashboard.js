document.addEventListener('DOMContentLoaded', () => {

    const startCameraButton = document.getElementById('startCameraButton');
    const closeCameraButton = document.getElementById('closeCameraButton');
    const cameraFeedContainer = document.getElementById('cameraFeedContainer');
    const cameraFeed = document.getElementById('cameraFeed');

    const predictedText = document.getElementById('predictedText');
    const finalText = document.getElementById('finalText');

    const startCapture = document.getElementById('startCapture');
    const stopCapture = document.getElementById('stopCapture');

    let predictionInterval;
    let currentAudio = null;
    let lastSpoken = "";

    // 🔊 TTS
    function speakText(text) {
        fetch("/text_to_speech", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        })
        .then(res => res.json())
        .then(data => {
            if (data.audio_url) {
                if (currentAudio) {
                    currentAudio.pause();
                    currentAudio.currentTime = 0;
                }
                currentAudio = new Audio(data.audio_url);
                currentAudio.play();
            }
        });
    }

    // ▶️ START CAMERA
    startCameraButton.addEventListener('click', () => {

        const feedUrl = startCameraButton.getAttribute('data-feed-url');

        cameraFeedContainer.style.display = 'block';
        cameraFeed.src = feedUrl;

        startCameraButton.style.display = 'none';
        closeCameraButton.style.display = 'inline-flex';

        // ✅ SHOW capture buttons ONLY NOW
        startCapture.style.display = 'inline-flex';
        stopCapture.style.display = 'inline-flex';

        predictionInterval = setInterval(async () => {
            const res = await fetch("/get_prediction");
            const data = await res.json();

            if (data.text && data.text.trim() !== "") {
                predictedText.innerText = "Live: " + data.text;

                if (data.text !== lastSpoken) {
                    speakText(data.text);
                    lastSpoken = data.text;
                }
            }
        }, 2000);
    });

    // ⛔ CLOSE CAMERA
    closeCameraButton.addEventListener('click', () => {

        cameraFeed.src = "";
        cameraFeedContainer.style.display = 'none';

        startCameraButton.style.display = 'inline-flex';
        closeCameraButton.style.display = 'none';

        // ❌ hide capture buttons again
        startCapture.style.display = 'none';
        stopCapture.style.display = 'none';

        clearInterval(predictionInterval);

        if (currentAudio) {
            currentAudio.pause();
            currentAudio.currentTime = 0;
        }

        predictedText.innerText = "";
        finalText.innerText = "";
        lastSpoken = "";
    });

    // 🟢 START CAPTURE
    startCapture.addEventListener("click", async () => {

        await fetch("/start_capture", { method: "POST" });

        finalText.innerText = "🟢 Capturing...";
        startCapture.classList.add("capturing");

        startCapture.disabled = true;
        stopCapture.disabled = false;
    });

    // 🔴 STOP CAPTURE
    stopCapture.addEventListener("click", async () => {

        const res = await fetch("/stop_capture", { method: "POST" });
        const data = await res.json();

        finalText.innerText = "Final: " + data.words;

        // remove glow
        startCapture.classList.remove("capturing");

        startCapture.disabled = false;
        stopCapture.disabled = true;

        speakText(data.words);
    });

});