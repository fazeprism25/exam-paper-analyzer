"""Provider-independent LLM backends.

llm_client.call_structured() is the ONLY thing in this package the rest of
the pipeline talks to (see exampapersorter/llm_client.py). Everything under
this package is an implementation detail of "how do we get one raw
completion out of a specific backend" -- callers never import from here
directly.
"""
