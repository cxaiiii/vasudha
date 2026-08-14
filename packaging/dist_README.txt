VASUDHA
An offline engineering assistant that computes its answers.
Built by Chaitanya, an independent developer.


GETTING STARTED

  1. Run Vasudha.exe
  2. It downloads its model once, about 2.3 GB. This takes a few minutes.
  3. Pick how you want it to talk to you.
  4. Ask it something.

After the first run it works with no internet at all.

Windows 10 or 11. Nothing else to install.


WHAT MAKES IT DIFFERENT

Ask it a calculation and it does not guess. It writes Python, runs it in a
real sandbox on your machine, and shows you the code and the output. Click
the card under any answer to see exactly what was computed.

Nothing you type leaves your computer. The one exception is web search, and
the interface tells you plainly whenever a query is sent.


WHAT IT IS BAD AT

  - It can pick a formula from a similar but wrong case. Check the equation
    it shows you, not just the number.
  - Research is weaker than calculation. Documents built without opening a
    real source are labelled "unverified" - believe that label.

The calculation is the part to trust. It shows its work so you can.


IF SOMETHING GOES WRONG

  Download fails
      The error appears on the setup screen with a Try again button.
      Downloads resume, so retrying does not start over.

  Already have the model file
      Click "I already have the file" and point it at your .gguf.

  Answers are slow
      Without a GPU it runs on the processor, which is much slower.
      Installing Ollama (ollama.com) lets Vasudha use your graphics card.
      It is detected automatically.

  Want a different personality
      Settings, then Personality. They are plain JSON files you can edit -
      there is an "Open personas folder" button.


Your chats, settings and downloaded model live in:
  %LOCALAPPDATA%\Vasudha

Model weights and details:
  https://huggingface.co/cxaiiii/vasudha-4b-v3-gguf

Apache 2.0 licensed.
