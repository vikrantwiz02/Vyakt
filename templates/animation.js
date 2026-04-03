const micBtn = document.getElementById("micBtn");

let mediaRecorder;
let audioChunks = [];

if (micBtn) {
    micBtn.addEventListener("click", async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: "audio/wav" });

                const formData = new FormData();
                formData.append("audio", audioBlob, "speech.wav");

                const response = await fetch("/speech_to_text", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();

                if (data.text) {
                    document.querySelector('input[name="sen"]').value = data.text;
                } else {
                    alert("Error: " + data.error);
                }
            };

            mediaRecorder.start();
            micBtn.innerText = "⏹️";

            setTimeout(() => {
                mediaRecorder.stop();
                micBtn.innerText = "🎤";
            }, 3000);

        } catch (err) {
            alert("Mic permission denied or not working");
            console.error(err);
        }
    });
}