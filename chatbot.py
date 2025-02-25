import aiml
import os

class Chatbot:
    def __init__(self, aiml_folder="AIML FILES"):
        self.kernel = aiml.Kernel()
        self.aiml_folder = aiml_folder
        self.load_aiml_files()

    def load_aiml_files(self):
        """Loads all AIML files from the specified directory."""
        if not os.path.exists(self.aiml_folder):
            print(f"Error: AIML directory '{self.aiml_folder}' not found.")
            return

        for filename in os.listdir(self.aiml_folder):
            if filename.endswith(".aiml"):
                file_path = os.path.join(self.aiml_folder, filename)
                self.kernel.learn(file_path)

    def preprocess_input(self, user_input):
        """Cleans and normalizes user input."""
        return user_input.strip().lower()

    def get_response(self, user_input):
        """Generates a response for a given user input with enhancements."""
        user_input = self.preprocess_input(user_input)

        if not user_input:
            return "I'm here to chat! Feel free to ask me anything."

        # Basic greeting detection
        greetings = ["hello", "hi", "hey", "good morning", "good evening"]
        if user_input in greetings:
            return "Hello! How can I assist you today?"

        response = self.kernel.respond(user_input)

        # Provide a fallback response if AIML doesn't match input
        return response if response else "I'm not sure I understand. Can you rephrase?"

# Example usage (for testing)
if __name__ == "__main__":
    bot = Chatbot()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        print("Chatbot:", bot.get_response(user_input))
