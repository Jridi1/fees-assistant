# Deploying to Hugging Face Spaces

The app runs as a Docker Space. The image contains the search index and the embedding model, so the Space needs no network for them at start-up, and waking from sleep takes seconds rather than minutes.

## What you need

- A Hugging Face account and a [write access token](https://huggingface.co/settings/tokens).
- Your Groq API key.
- The public address of your portfolio, for example `https://you.pages.dev`.

## 1. Check the image locally (optional but worth it)

```bash
docker build -t fees-assistant .
docker run --rm -p 7860:7860 -e GROQ_API_KEY=your-key fees-assistant
```

Open http://localhost:7860/. The first build takes several minutes (downloads and indexing); later builds reuse cached layers.

## 2. Create the Space

1. huggingface.co/new-space, choose **Docker** as the SDK and **CPU basic** hardware (free), and pick a name such as `fees-assistant`.
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
