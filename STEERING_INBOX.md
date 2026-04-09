(no pending instructions)

<!--
HOW TO USE THIS FILE — read carefully before editing on your phone.

This is the INBOX for the `scaling-ops-steer` remote trigger. To send a
steering instruction from your phone:

  1. Open this file on github.com/lucasperrier/scaling-operator-learning
     (the GitHub web editor works on mobile).
  2. Replace the line `(no pending instructions)` with your instruction(s).
     Use a numbered list. Be specific. Mention exact files when relevant.
  3. Commit the change directly to `main` from the GitHub web UI.
  4. Open claude.ai/code on your phone.
  5. Find the trigger named `scaling-ops-steer` and click "Run trigger".
  6. The agent will git pull, read this file, execute the instructions,
     append a record to STEERING_LOG.md, clear this file back to
     "(no pending instructions)", and push.
  7. Read the agent's report in claude.ai/code.

INSTRUCTION CATEGORIES (the agent recognizes these):

  - **Answer a pending decision** the laptop session is blocked on.
    The agent writes the answer to DECISIONS_RESOLVED.md and the laptop
    reads it on its next git pull.
      Example: "Question 3 in DECISIONS_PENDING.md: yes, drop Darcy from
      the main text and move it to an appendix."

  - **Small repo edit**: a config tweak, a doc fix, a one-line code
    change. Don't ask for big refactors here — those are the laptop's
    job.
      Example: "In configs/burgers_operator.yaml, set
      train_resolutions to [32, 64, 128, 256, 512]."

  - **Note for the laptop session**: the agent appends to
    NOTES_FROM_REMOTE.md.
      Example: "Note for laptop: prefer scipy over jax for any new
      analysis code; scipy is already installed."

  - **Reframe redirect**: a directive that should change how the laptop
    is doing the reframe. The agent appends to REFRAME_DIRECTIVES.md.
      Example: "Skip Phase 6 (APPENDIX_LAW_FITS.md). The held-out
      contest result alone is enough fragility evidence."

  - **Status query**: just ask the agent, it reports back without
    writing to disk.
      Example: "What is the laptop session currently blocked on? Is
      anything in DECISIONS_PENDING.md? What was the last commit?"

WHAT THE AGENT WILL REFUSE:

  - Anything destructive: force-push, reset --hard, branch deletion,
    skipping hooks.
  - Edits under runs/, runs_ablation_*/, results/, or .venv/.
  - Doing the next phase of the reframe itself ("write PROTOCOL_V2.md
    for me"). That is the laptop session's job. If you want to redirect
    the laptop, write a directive instead.
  - Anything that would conflict with uncommitted in-flight work the
    laptop has.

When you're done with this session and don't need steering anymore, you
can leave this file as "(no pending instructions)" — it costs nothing.
-->
