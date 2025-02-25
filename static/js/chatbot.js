document.addEventListener("DOMContentLoaded", function () {
    const chatButton = document.getElementById("chat-with-us");
    const chatContainer = document.querySelector(".chat-container");
    const closeButton = document.querySelector(".close-btn");
    const chatBody = document.getElementById("chat-body");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");

    // Background Blur
    const blurOverlay = document.querySelector(".blur-overlay");

    // ✅ Hide chatbox on page load (ensuring no glitches)
    chatContainer.style.display = "none";

    // Open Chatbot
    chatButton.addEventListener("click", function () {
        chatContainer.classList.add("active");
        chatContainer.style.display = "flex";  // ✅ Shows chatbox only when clicked
        blurOverlay.classList.add("show");
    });

    // Close Chatbot
    closeButton.addEventListener("click", function () {
        chatContainer.classList.remove("active");
        setTimeout(() => {
            chatContainer.style.display = "none"; // ✅ Hides chatbox properly
        }, 300);
        blurOverlay.classList.remove("show");
    });

    // Send Message
    function sendMessage() {
        let message = userInput.value.trim();
        if (message === "") return;

        appendMessage("You", message);
        userInput.value = "";

        fetch("/get_response", {
            method: "POST",
            body: JSON.stringify({ message: message }),
            headers: { "Content-Type": "application/json" }
        })
        .then(response => response.json())
        .then(data => appendMessage("Chatbot", data.response))
        .catch(error => console.error("Error:", error));
    }

    // Append Message
    function appendMessage(sender, message) {
        let messageDiv = document.createElement("div");
        messageDiv.classList.add(sender === "You" ? "user-message" : "bot-message");
        messageDiv.textContent = `${sender}: ${message}`;
        chatBody.appendChild(messageDiv);
        chatBody.scrollTop = chatBody.scrollHeight;
    }

    // Event Listeners
    sendBtn.addEventListener("click", sendMessage);
    userInput.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
            event.preventDefault();
            sendMessage();
        }
    });
});
