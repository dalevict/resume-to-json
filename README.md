# resume-to-json
Parses resume into JSON using LLMs for more accuracy than just keywords

# launch local server
in llama binaries `/llama-b9670` or equivalent directory: 
`./llama-server -hf mradermacher/mistral7b_text_to_json_v3.1-GGUF:Q4_K_M -c 8192 --port 8080`
* `./llama-server`: Executes the local server binary inside your current folder.

* `-hf`: Automatically pulls down the `Q4_K_M` weight file directly from the repo link you provided.

* `-c 8192`: Expands the model's context loop to 8k tokens to handle larger resume parsing strings smoothly.

# launch api
once the llama server is up, in the main directory:
`main.py`