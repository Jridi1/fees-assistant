# Deploying to Hugging Face Spaces

The app runs as a Docker Space. The image contains the search index and the embedding model, so the Space needs no network for them at start-up, and waking from sleep takes seconds rather than minutes.

## What you need

- A Hugging Face account **on the PRO plan ($9/month)**. Hugging Face's docs say: "Gradio and Docker Spaces run on compute and require a paid plan to create: PRO for personal accounts." The CPU Basic *hardware* has no hourly cost, but a free account cannot create a Docker Space. (Static Spaces are free, but cannot run this app.) If you would rather not pay, see "Other ways to host it" at the end.
- A [write access token](https://huggingface.co/settings/tokens).
- Your Groq API key.
- The public address of your portfolio, for example `https://you.pages.dev`.

## 1. Check the image locally (optional but worth it)

```bash
docker build -t fees-assistant .
docker run --rm -p 7860:7860 -e GROQ_API_KEY=your-key fees-assistant
```

Open http://localhost:7860/. The first build takes several minutes (downloads and indexing); later builds reuse cached layers.

## 2. Create the Space

1. huggingface.co/new-space, choose **Docker** as the SDK and **CPU basic** hardware (no hourly cost; the PRO plan is what pays for creating the Space), and pick a name such as `fees-assistant`. Set visibility to **Public**.
2. In the Space's **Settings**, add:
   - **Secret** `GROQ_API_KEY` (your key).
   - **Variable** `CORS_ORIGINS`, a JSON list of the sites allowed to embed the chat page:
     `["https://you.pages.dev","https://huggingface.co"]`
     Include `https://huggingface.co` so the page also works inside the Space's own huggingface.co view.
   - **Variable** `TRUSTED_PROXY_HOPS`, set in step 4.

## 3. Push the code

From the `rev/` folder. Keep GitHub as your main remote and add the Space as a second one:

```bash
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
git push space main
```

When asked for a password, paste your Hugging Face token. The Space builds automatically (watch the **Logs** tab); when it says **Running**, the address is `https://<your-username>-<space-name>.hf.space`.

> **Careful:** `README.md` now says `sdk: docker`. If your old Gradio Space (`revolut-fees-assistant`) is connected to the same repository, it will stop building. Create a *new* Space for this app, as above.

## 4. Set `TRUSTED_PROXY_HOPS`

Hugging Face puts a proxy in front of your app, so the socket address is the proxy's, not the visitor's. Left at `0`, every visitor would share one rate limit. Set it to `1` and check that visitors really are told apart:

1. Send more questions than `RATE_LIMIT_PER_MINUTE` (default 8) in a minute from your laptop. The extra ones should be refused with "You're asking questions quickly".
2. While that is happening, ask a question from your phone on mobile data. It should still be answered. If it is refused too, the proxy count is too low: everyone shares one address.
3. Try to cheat the limit from the laptop by sending a forged header, so the limit resets:

   ```bash
   curl -s -X POST https://<your-username>-<space-name>.hf.space/chat \
     -H "Content-Type: application/json" -H "X-Forwarded-For: 1.2.3.4" \
     -d '{"session_id":"probe","question":"replacement card"}'
   ```

   Repeat with a different forged value each time. If you can go past the limit, the count is too high: the app is trusting an address the client wrote. Lower it.

## 5. Connect the portfolio

In `site/index.html`, set `DEMO_URL` at the top of the script to your Space address, then upload the site again to Cloudflare Pages. The "Try it live" button opens the chat in a modal, and shows a "waking up" message while a sleeping Space starts.

## Things to know

- **Free Spaces sleep** after a period without visitors and wake on the next request. The portfolio warms the Space as soon as the page loads, and the modal shows progress while it wakes.
- **Nothing persists** across restarts: conversation memory is in RAM and the index is rebuilt from the image. That is intended.
- **The daily budget is a guess.** Check your Groq usage after the first day and adjust `DAILY_REQUEST_BUDGET` and `GLOBAL_RATE_LIMIT_PER_MINUTE`.
- **Updating:** `git push space main` rebuilds. If you change the PDFs or the chunk settings the index is rebuilt during the image build, so the running Space never serves a stale index.

## Other ways to host it

The app is a plain Docker image that listens on one port (7860 by default), so any host that runs containers can serve it. What it needs: about 2 GB of memory (PyTorch and the embedding model), the environment variables from the README, and no persistent disk (the index is baked into the image). Options, with the trade-offs:

- **Hugging Face Spaces with PRO ($9/month).** Simplest, and everything above applies as written.
- **A free-tier container host** (for example Google Cloud Run, which has a free allowance but needs a billing account with a card on file). Check the memory limit of the free tier before choosing: a 512 MB instance is too small for this app. Set the container port to 7860 and add the same variables. `TRUSTED_PROXY_HOPS` must be worked out again for that host's proxy.
- **No live hosting.** Keep `DEMO_URL` empty (the "Try it live" buttons then stay hidden) and show a recorded demo instead.
- **On demand from your own computer**, exposed through a tunnel, for an interview or a call. Free, but only while your machine is on.

## Run it from your laptop, on demand (no hosting cost)

Good for an interview or a call: you start the app, expose it through a free Cloudflare quick tunnel, and share the address it prints. The address **changes every run**, so do not put it in the portfolio: leave `DEMO_URL` empty and paste the link in the chat or email instead.

**Once:** install the tunnel tool, and build the index *before* the call (starting the server after a settings change rebuilds it, which takes minutes and a lot of memory).

```powershell
winget install --id Cloudflare.cloudflared     # or download cloudflared-windows-amd64.exe from
                                               # github.com/cloudflare/cloudflared/releases
python -m feesbot ingest                       # only needs to be done again if the PDFs or settings change
```

**Each time** (two terminals, both in `rev/`, the first with the virtual environment active):

```powershell
# terminal 1: the app. Optionally cap the day's budget for a private demo.
$env:DAILY_REQUEST_BUDGET = 60
python -m feesbot serve                        # wait for "Ready."

# terminal 2: the tunnel
cloudflared tunnel --url http://127.0.0.1:8000 # prints https://<random-words>.trycloudflare.com
```

Open the printed address yourself first and ask one question. Share it. Press Ctrl+C in both terminals when you are done.

What was checked while writing this (with a throwaway server, not your real one):

- The chat page loads through the tunnel in Chrome, the status lines arrive one by one, and the answer renders. Cloudflare's own documentation says quick tunnels do **not** support server-sent events, yet streaming worked in this test. Treat that as "works today, not guaranteed": if the page ever shows all its progress at once or stalls, that is the likely reason, and the fallback is to share your screen.
- **`TRUSTED_PROXY_HOPS` stays at 0.** The tunnel connects from your own machine, and uvicorn already takes the visitor's address from the header the tunnel adds. A forged `X-Forwarded-For` was ignored.
- Anyone with the link can use your Groq quota until you stop the tunnel. The per-visitor and daily limits still apply; lowering `DAILY_REQUEST_BUDGET` as above keeps a demo small.
- The tunnel and the server stop when your laptop sleeps or you close the terminals. Cloudflare describes quick tunnels as meant for testing and development, not production.
