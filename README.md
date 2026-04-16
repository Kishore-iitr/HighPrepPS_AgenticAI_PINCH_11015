# agentSrishti

VIDEO EXPLAINATION: [LINK](https://iitracin-my.sharepoint.com/:v:/g/personal/kishore_s_ph_iitr_ac_in/IQB-N0MSVtsySabe1uWdcKodAYmQbQGOOtMtkM4k8rH3yGs?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=lOdviK)

REPORT: [LINK](https://iitracin-my.sharepoint.com/:b:/g/personal/kishore_s_ph_iitr_ac_in/IQB02Q5Gyx8aSLcgqHBhwgK8AT_FX_Sp3mjjZfxHctE9wKc?e=8AkzZj)

# Agentic Event Planner

This workspace now includes a hostable Streamlit website that turns the notebook flows into an interactive event-planning app.

```
Agentic-Event-Planner
├── 📁 data/                     # Raw datasets (events, sponsors, speakers, etc.)
│   ├── events_v2.csv
│   ├── sponsors_v2.csv
│   ├── speakers_v2.csv
│   ├── venues.csv
│   ├── communities.csv
│   ├── contacts.csv
│   └── ...
│
├── 📁 vector_db/                # Base vector database (general embeddings)
├── 📁 sponsor_vector_db/        # Sponsor-specific embeddings
├── 📁 speaker_vector_db/        # Speaker-specific embeddings
├── 📁 exhibitor_vector_db/      # Exhibitor-specific embeddings
├── 📁 venue_vector_db/          # Venue-specific embeddings
├── 📁 community_vector_db/      # Community/GTM embeddings
├── 📁 event_ops_vector_db/      # Event operations embeddings
├── 📁 pricing_vector_db/        # Pricing intelligence embeddings
├── 📁 contact_vector_db/        # Outreach/contact embeddings
│
├── 📄 main.py                   # Core LangGraph orchestration (multi-agent system)
├── 📄 pricing_engine.py         # ML-based pricing engine
│
├── 📄 README.md                 # Project documentation
├── 📄 .gitignore                # Ignored files and folders
│
└── 📁 .streamlit/ (optional)    # Streamlit config (if used)
    └── config.toml
```


## Run locally

1. Install dependencies from `requirements.txt`.
2. Set the environment variables you want to use:
   - `OPENROUTER_API_KEY`
   - `TAVILY_API_KEY`
   - `LANGSMITH_API_KEY` or `LANGCHAIN_API_KEY` if you want tracing
  
   <img width="1480" height="499" alt="image" src="https://github.com/user-attachments/assets/14d3df79-e8a1-4165-9f88-3923fa407a14" />

4. Launch the site:

```bash
streamlit run main.py
```

## What the app includes

- Sponsor, speaker, exhibitor, venue, community, pricing, event ops, and outreach workflows
- PricingEngine-backed forecasting and tier optimization
- Interactive ticket tier simulator with promo discount controls
- Break-even analysis with sliders for venue, speaker, ops, marketing, capacity, and ticket price
- Chroma vector database loading from the existing data directories
- Optional Tavily web enrichment and LangSmith tracing support
- Outreach drafting with Gmail send support when credentials are available


  <img width="1445" height="645" alt="grpah_basic" src="https://github.com/user-attachments/assets/bf292a4a-9c0d-4e61-b52f-19a90af5cc3c" />



