import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class PricingEngine:
    """ML engine for ticket pricing and footfall prediction."""

    def __init__(
        self,
        events_path: str = "data/events_v2.csv",
        venues_path: str = "data/venues.csv",
        pricing_tiers_path: str = "data/pricing_tiers.csv",
        pricing_vector_db_path: str = "./pricing_vector_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        random_state: int = 42,
    ) -> None:
        self.events_path = events_path
        self.venues_path = venues_path
        self.pricing_tiers_path = pricing_tiers_path
        self.pricing_vector_db_path = pricing_vector_db_path
        self.embedding_model = embedding_model
        self.random_state = random_state

        self.events_df: Optional[pd.DataFrame] = None
        self.venues_df: Optional[pd.DataFrame] = None
        self.pricing_df: Optional[pd.DataFrame] = None

        self.event_level_df: Optional[pd.DataFrame] = None
        self.tier_level_df: Optional[pd.DataFrame] = None

        self.attendance_model: Optional[Pipeline] = None
        self.conversion_model: Optional[Pipeline] = None

        self.price_min: Optional[float] = None
        self.price_max: Optional[float] = None
        self.default_city: str = ""
        self.default_country: str = ""
        self.default_category: str = ""
        self.default_audience_size: int = 0

        self.model_metrics: Dict[str, float] = {}

        self.embedding_fn: Optional[HuggingFaceEmbeddings] = None
        self.pricing_vectordb: Optional[Chroma] = None

    def load_data(self) -> None:
        self.events_df = pd.read_csv(self.events_path)
        self.venues_df = pd.read_csv(self.venues_path)
        self.pricing_df = pd.read_csv(self.pricing_tiers_path)

    def _build_event_to_venue_map(self) -> pd.DataFrame:
        mappings: List[Dict[str, Any]] = []
        for _, row in self.venues_df.iterrows():
            events = [e.strip() for e in str(row.get("past_events", "")).split(",") if e.strip()]
            for event_name in events:
                mappings.append(
                    {
                        "event_name": event_name,
                        "venue_name": row.get("venue_name", ""),
                        "capacity": row.get("capacity", np.nan),
                        "estimated_cost": row.get("estimated_cost", np.nan),
                        "venue_city": row.get("city", ""),
                        "venue_country": row.get("country", ""),
                    }
                )
        return pd.DataFrame(mappings)

    @staticmethod
    def _first_non_empty(series: pd.Series, fallback: str = "") -> str:
        cleaned = series.dropna().astype(str).str.strip()
        cleaned = cleaned[cleaned != ""]
        if cleaned.empty:
            return fallback
        modes = cleaned.mode()
        if not modes.empty:
            return str(modes.iloc[0])
        return str(cleaned.iloc[0])

    def preprocess(self) -> None:
        if self.events_df is None or self.venues_df is None or self.pricing_df is None:
            raise ValueError("Run load_data() before preprocess().")

        events = self.events_df.copy()
        venues = self.venues_df.copy()
        pricing = self.pricing_df.copy()

        events["attendance"] = pd.to_numeric(events["attendance"], errors="coerce")
        events["year"] = pd.to_numeric(events["year"], errors="coerce")

        venues["capacity"] = pd.to_numeric(venues["capacity"], errors="coerce")
        venues["estimated_cost"] = pd.to_numeric(venues["estimated_cost"], errors="coerce")

        pricing["price"] = pd.to_numeric(pricing["price"], errors="coerce")
        pricing["tickets_sold"] = pd.to_numeric(pricing["tickets_sold"], errors="coerce")
        pricing["total_available"] = pd.to_numeric(pricing["total_available"], errors="coerce")
        pricing["conversion_rate"] = pd.to_numeric(pricing["conversion_rate"], errors="coerce")

        ticket_ranges = events["ticket_price_range"].astype(str).str.split("-", n=1, expand=True)
        events["ticket_price_min"] = pd.to_numeric(ticket_ranges[0], errors="coerce")
        events["ticket_price_max"] = pd.to_numeric(ticket_ranges[1], errors="coerce")
        events["price_mid"] = (events["ticket_price_min"] + events["ticket_price_max"]) / 2.0

        tier_event = (
            pricing.groupby("event_name", as_index=False)
            .agg(
                avg_tier_price=("price", "mean"),
                tier_revenue=("price", lambda x: float(np.sum(x * pricing.loc[x.index, "tickets_sold"]))),
                tickets_sold_total=("tickets_sold", "sum"),
                total_available_total=("total_available", "sum"),
                conversion_mean=("conversion_rate", "mean"),
            )
        )

        event_venue_map = self._build_event_to_venue_map()

        merged = events.merge(tier_event, on="event_name", how="left")
        merged = merged.merge(event_venue_map, on="event_name", how="left")

        merged["capacity"] = merged["capacity"].fillna(merged.groupby("city")["capacity"].transform("median"))
        merged["capacity"] = merged["capacity"].fillna(merged["attendance"])

        merged["fill_rate"] = (merged["attendance"] / merged["capacity"]).clip(lower=0.0, upper=1.0)
        merged["revenue"] = merged["tier_revenue"].fillna(merged["attendance"] * merged["price_mid"])
        merged["demand_indicator"] = (
            merged["tickets_sold_total"].fillna(0) / merged["total_available_total"].replace(0, np.nan)
        ).fillna(merged["fill_rate"])

        merged["category"] = merged["category"].fillna("Unknown")
        merged["city"] = merged["city"].fillna("Unknown")
        merged["country"] = merged["country"].fillna("Unknown")

        merged = merged.dropna(subset=["attendance", "price_mid", "capacity"]).copy()

        self.event_level_df = merged

        self.default_category = self._first_non_empty(events["category"])
        self.default_city = self._first_non_empty(events["city"])
        self.default_country = self._first_non_empty(events["country"])
        self.default_audience_size = int(round(float(events["attendance"].median()))) if events["attendance"].notna().any() else 0

        tier_level = pricing.merge(events[["event_name", "category", "city", "country"]], on="event_name", how="left")
        tier_level["category"] = tier_level["category"].fillna("Unknown")
        tier_level["city"] = tier_level["city"].fillna("Unknown")
        tier_level["country"] = tier_level["country"].fillna("Unknown")
        tier_level = tier_level.dropna(subset=["price", "conversion_rate"]).copy()
        self.tier_level_df = tier_level

        if pricing["price"].notna().any():
            self.price_min = float(pricing["price"].quantile(0.05))
            self.price_max = float(pricing["price"].quantile(0.95))
        else:
            self.price_min = float(events["price_mid"].median()) if events["price_mid"].notna().any() else 0.0
            self.price_max = float(events["price_mid"].max()) if events["price_mid"].notna().any() else self.price_min

        self._build_pricing_rag_store()

    def _build_pricing_rag_store(self) -> None:
        docs: List[Document] = []
        if self.event_level_df is None or len(self.event_level_df) == 0:
            return

        for _, row in self.event_level_df.iterrows():
            text = (
                f"{row['event_name']} | {row['category']} | {row['city']}, {row['country']} | "
                f"attendance={int(row['attendance'])} | capacity={int(row['capacity'])} | "
                f"price_mid={row['price_mid']:.2f} | conversion={row.get('conversion_mean', np.nan):.3f} | "
                f"revenue={row.get('revenue', np.nan):.2f}"
            )
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "event_name": str(row.get("event_name", "")),
                        "category": str(row.get("category", "")),
                        "city": str(row.get("city", "")),
                        "country": str(row.get("country", "")),
                    },
                )
            )

        self.embedding_fn = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.pricing_vectordb = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding_fn,
            persist_directory=self.pricing_vector_db_path,
        )
        self.pricing_vectordb.persist()

    def train_model(self) -> None:
        if self.event_level_df is None or self.tier_level_df is None:
            raise ValueError("Run preprocess() before train_model().")

        # Attendance model
        att_df = self.event_level_df.copy()
        X_att = att_df[["price_mid", "city", "category", "capacity"]]
        y_att = att_df["attendance"].astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X_att, y_att, test_size=0.25, random_state=self.random_state
        )

        pre_att = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), ["price_mid", "capacity"]),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    ["city", "category"],
                ),
            ]
        )

        att_model = Pipeline(
            steps=[
                ("pre", pre_att),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=self.random_state,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        )

        att_model.fit(X_train, y_train)
        att_pred = att_model.predict(X_test)

        # Precision proxy for regression: pct predictions within 10% of true value.
        rel_error = np.abs(att_pred - y_test.values) / np.maximum(np.abs(y_test.values), 1.0)
        att_precision = float(np.mean(rel_error <= 0.10))

        self.attendance_model = att_model

        # Conversion model
        conv_df = self.tier_level_df.copy()
        X_conv = conv_df[["price", "tier_name", "city", "category"]]
        y_conv = conv_df["conversion_rate"].clip(lower=0.0, upper=1.0)

        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            X_conv, y_conv, test_size=0.25, random_state=self.random_state
        )

        pre_conv = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), ["price"]),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    ["tier_name", "city", "category"],
                ),
            ]
        )

        conv_model = Pipeline(
            steps=[
                ("pre", pre_conv),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=250,
                        random_state=self.random_state,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        )

        conv_model.fit(Xc_train, yc_train)
        conv_pred = conv_model.predict(Xc_test)
        conv_rel_error = np.abs(conv_pred - yc_test.values) / np.maximum(np.abs(yc_test.values), 1e-3)
        conv_precision = float(np.mean(conv_rel_error <= 0.10))

        self.conversion_model = conv_model

        self.model_metrics = {
            "attendance_r2": float(r2_score(y_test, att_pred)),
            "attendance_rmse": float(np.sqrt(mean_squared_error(y_test, att_pred))),
            "attendance_mae": float(mean_absolute_error(y_test, att_pred)),
            "attendance_precision_within_10pct": att_precision,
            "conversion_r2": float(r2_score(yc_test, conv_pred)),
            "conversion_rmse": float(np.sqrt(mean_squared_error(yc_test, conv_pred))),
            "conversion_mae": float(mean_absolute_error(yc_test, conv_pred)),
            "conversion_precision_within_10pct": conv_precision,
        }

    def _select_venue_capacity(self, city: str, location: str) -> Tuple[int, str]:
        venues = self.venues_df.copy()
        city_match = venues[venues["city"].astype(str).str.lower() == city.lower()]
        country_match = venues[venues["country"].astype(str).str.lower() == location.lower()]

        candidates = city_match if len(city_match) > 0 else country_match
        if len(candidates) == 0:
            candidates = venues

        row = candidates.sort_values("capacity", ascending=True).iloc[-1]
        return int(row["capacity"]), str(row["venue_name"])

    def _predict_conversion(self, price: float, tier_name: str, city: str, category: str) -> float:
        if self.conversion_model is None:
            raise ValueError("Run train_model() before _predict_conversion().")

        sample = pd.DataFrame(
            [
                {
                    "price": float(price),
                    "tier_name": tier_name,
                    "city": city,
                    "category": category,
                }
            ]
        )
        conv = float(self.conversion_model.predict(sample)[0])
        return float(np.clip(conv, 0.01, 0.99))

    def _predict_attendance(self, price: float, city: str, category: str, capacity: int) -> float:
        if self.attendance_model is None:
            raise ValueError("Run train_model() before _predict_attendance().")

        sample = pd.DataFrame(
            [
                {
                    "price_mid": float(price),
                    "city": city,
                    "category": category,
                    "capacity": float(capacity),
                }
            ]
        )
        return float(max(self.attendance_model.predict(sample)[0], 0.0))

    def generate_tiers(self, base_price: float, city: str, category: str) -> Dict[str, Dict[str, float]]:
        early = max(self.price_min, base_price * 0.70)
        regular = base_price
        vip = min(self.price_max * 1.75, base_price * 2.0)

        return {
            "early_bird": {
                "price": float(early),
                "expected_conversion": self._predict_conversion(early, "early_bird", city, category),
            },
            "regular": {
                "price": float(regular),
                "expected_conversion": self._predict_conversion(regular, "regular", city, category),
            },
            "vip": {
                "price": float(vip),
                "expected_conversion": self._predict_conversion(vip, "vip", city, category),
            },
        }

    def optimize_price(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        city = str(input_data.get("city") or self.default_city or "")
        location = str(input_data.get("location") or self.default_country or self.default_city or "")
        category = str(input_data.get("category") or self.default_category or "")
        target_audience_raw = input_data.get("audience_size")
        target_audience = int(target_audience_raw if target_audience_raw not in (None, "") else self.default_audience_size or 0)

        capacity, venue_name = self._select_venue_capacity(city, location)

        low = float(self.price_min if self.price_min is not None else 0.0)
        high = float(self.price_max if self.price_max is not None else low)
        if high <= low:
            high = low + max(low * 0.5, 1.0)
        prices = np.linspace(low, high, 40)

        scenarios: List[Dict[str, float]] = []
        for p in prices:
            att = self._predict_attendance(float(p), city, category, capacity)
            conv = self._predict_conversion(float(p), "regular", city, category)
            adj_att = min(capacity, max(0.0, att * (0.65 + 0.35 * conv)))
            revenue = float(adj_att * p)
            scenarios.append(
                {
                    "price": float(p),
                    "predicted_attendance": float(adj_att),
                    "revenue": revenue,
                    "conversion_regular": float(conv),
                }
            )

        scenarios_df = pd.DataFrame(scenarios).sort_values("revenue", ascending=False).reset_index(drop=True)
        best = scenarios_df.iloc[0]

        best_price = float(best["price"])
        expected_attendance = int(round(min(best["predicted_attendance"], capacity)))

        tiers = self.generate_tiers(best_price, city, category)

        conv_sum = sum(v["expected_conversion"] for v in tiers.values())
        for tier_name, data in tiers.items():
            share = data["expected_conversion"] / conv_sum if conv_sum > 0 else 1.0 / 3.0
            sold = int(round(expected_attendance * share))
            data["expected_tickets_sold"] = sold
            data["expected_revenue"] = float(sold * data["price"])

        total_revenue = float(sum(v["expected_revenue"] for v in tiers.values()))
        fill_rate = float(expected_attendance / capacity) if capacity > 0 else 0.0

        rag_context: List[str] = []
        if self.pricing_vectordb is not None:
            query = f"{category} conference pricing in {city} {location} attendance conversion"
            docs = self.pricing_vectordb.similarity_search(query, k=3)
            rag_context = [d.page_content for d in docs]

        return {
            "base_price": int(round(best_price)),
            "tiers": tiers,
            "expected_attendance": expected_attendance,
            "venue_capacity": int(capacity),
            "venue_name": venue_name,
            "target_audience": target_audience,
            "revenue": float(round(total_revenue, 2)),
            "fill_rate": float(round(fill_rate, 4)),
            "insights": [
                f"Base price optimized over historical range [{low:.0f}, {high:.0f}]",
                f"Capacity constraint applied using venue: {venue_name}",
                "Attendance adjusted with learned conversion response",
            ],
            "top_scenarios": scenarios_df.head(3).to_dict(orient="records"),
            "model_metrics": self.model_metrics,
            "rag_context": rag_context,
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.load_data()
        self.preprocess()
        self.train_model()
        return self.optimize_price(input_data)
