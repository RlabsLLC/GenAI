# GenAI

Custom GGUF AI models built completely from scratch. These models dynamically grow in size during training and come in various parameter sizes.

## Latest Version: GenA1_63M.gguf

Release Notes & Capabilities

- Interactive: Can answer basic questions and recognize conversational context.

- Formatting: Understands and generates basic Markdown formatting (like bolding and bullet points).

- Mini-Sized: Incredibly lightweight at ~250MB (63 Million Parameters).

- Custom Architecture: Uses a custom dynamically-grown transformer architecture **(Note: This means it does not run in standard Ollama environments).**

- Contextual: Has a growing understanding of the user's prompt and intent.

## Example Chat (GenA1 18M)

```
You: Hi

GenA1: ng to emulate the reasonsebing more efficient, scausidg. Thim triight relesting might say,  It s a integroenuity en some of a welcoming overwhat."
* **Agreement (Possible):** It s your day going? 
* "Hi! What are you up to this what are you re working on.
* **Contial Opening:** "I'm s a, "Hey! How's your day going?"
* **How are you today?:** "How are you doing today?"
* **Simple Great for almo, in describles. The Symbol Computer, Grom the development of symbolic AI programs, allowing researchers to understanding! How are you understanding the "Symbol Cuming and what forhere's a really insitions:** Trouble?   Imple & Friendly (Great for flow, engage and natural repoy of AI possibilities   and agreement, but it!)
**7.  BHellow are you conversation.)

**3. Freet (Gentle fraportate, and The "Let's bras brad understanding of how they milestay, situations   for from texting to quick cha
```

## How to Run the Model

Because this model uses a custom architecture, it requires our custom Python loader (Loader.py) to run.

1. Install Requirements

Make sure you have Python installed, then install the required dependencies `pip install gguf torch`


2. Download the Files

To chat with the AI, you need three files saved in the same folder:

- The model weights (.gguf file)

- The model's alphabet (_vocab.json file)

- The custom loader script (Loader.py)

If you want to download Loader.py directly from the terminal, use one of the commands below:

Mac/Linux:

```
curl -O [https://raw.githubusercontent.com/RlabsLLC/GenAI/main/Loader.py](https://raw.githubusercontent.com/RlabsLLC/GenAI/main/Loader.py)
```


Windows (PowerShell):

```
Invoke-WebRequest -Uri "[https://raw.githubusercontent.com/RlabsLLC/GenAI/main/Loader.py](https://raw.githubusercontent.com/RlabsLLC/GenAI/main/Loader.py)" -OutFile "Loader.py"
```


3. Start Chatting!

Open your terminal in the folder where you downloaded the files and run:

`python Loader.py`


*When prompted, just type the name of the .gguf file you downloaded, press enter, and say hello! Models are in the /Models Dir*
