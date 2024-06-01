class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button'),
            voiceButton: document.getElementById('voiceButton'),
            voiceResponse: document.getElementById('voiceResponse')
        }

        this.state = false;
        this.messages = [];
        this.speechSynthesis = window.speechSynthesis;
        this.speechUtterance = new SpeechSynthesisUtterance();
        this.voiceButtonEnabled = false;
        this.isSpeaking = false;
    }

    display() {
        const { openButton, chatBox, sendButton, voiceButton } = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox));

        sendButton.addEventListener('click', () => this.onSendButton(chatBox));

        voiceButton.addEventListener('click', () => this.toggleVoice());

        voiceButton.style.color = 'white';

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox);
            }
        });
    }

    toggleState(chatbox) {
        this.state = !this.state;

        // show or hide the box
        if (this.state) {
            chatbox.classList.add('chatbox--active');
        } else {
            chatbox.classList.remove('chatbox--active');
        }
    }

    toggleVoice() {
        // Toggle the voice speaking state
        this.isSpeaking = !this.isSpeaking;

        // Toggle the class of the microphone icon element
        const voiceButton = this.args.voiceButton;
        voiceButton.classList.toggle('fa-microphone-slash', this.isSpeaking);

        if (this.isSpeaking) {
            // If voice is enabled, speak the most recent chatbot response
            this.speakMostRecentChatbotResponse();
        } else {
            // If voice is disabled, stop speaking
            this.stopSpeaking();
        }
    }

    speakMostRecentChatbotResponse() {
        if (this.messages.length > 0) {
            const lastChatbotResponse = this.messages
                .slice()
                .reverse()
                .find((message) => message.name === "Sam");

            if (lastChatbotResponse) {
                // Speak the most recent chatbot response
                this.speechUtterance.text = lastChatbotResponse.message;
                this.speechSynthesis.speak(this.speechUtterance);
            }
        }
    }


    speakResponse() {
        if (this.messages.length > 0) {
            const lastMessage = this.messages[this.messages.length - 1];
            const textToSpeak = lastMessage.message;
            this.speechUtterance.text = textToSpeak;
            this.speechSynthesis.speak(this.speechUtterance);
        }
    }

    stopSpeaking() {
        this.speechSynthesis.cancel();
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", message: text1 }
        this.messages.push(msg1);

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
        .then(r => r.json())
        .then(r => {
            let msg2 = { name: "Sam", message: r.answer };
            this.messages.push(msg2);
            this.updateChatText(chatbox);
            textField.value = '';

            if (this.voiceButtonEnabled) {
                this.speakResponse();
            }
        }).catch((error) => {
            console.error('Error:', error);
            this.updateChatText(chatbox);
            textField.value = '';
        });
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item, index) {
            if (item.name === "Sam") {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>'
            } else {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
}

const chatbox = new Chatbox();
chatbox.display();