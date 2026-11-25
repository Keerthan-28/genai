"""Central deterministic metric registry for VeriReport AI."""

from __future__ import annotations

from backend.schemas import MetricDefinition


UNSUPPORTED_METRIC_KEYWORDS: tuple[str, ...] = (
    "profit",
    "margin",
    "ebitda",
    "roi",
    "marketing",
    "customer satisfaction",
    "csat",
    "nps",
    "inventory",
    "stock level",
    "headcount",
    "churn",
    "conversion rate",
    "click",
)


class MetricRegistry:
    """Registry of supported retail analytics metrics, rules, and tolerances."""

    def __init__(self) -> None:
        self._metrics: dict[str, MetricDefinition] = {
            m.metric_id: m for m in self._build_default_metrics()
        }

    @staticmethod
    def _build_default_metrics() -> list[MetricDefinition]:
        return [
            MetricDefinition(
                metric_id="total_revenue",
                metric_name="Total Revenue",
                description="Total revenue computed under the active transaction policy.",
                required_columns=["Quantity", "UnitPrice", "InvoiceDate"],
                formula="SUM(Quantity * UnitPrice) under active transaction policy",
                supported_operations=["value", "growth_pct", "share_pct"],
                applicable_filters=["period", "product", "country"],
                unit="currency",
                validation_rules=["Quantity and UnitPrice must be numeric; UnitPrice > 0"],
                abs_tolerance=1.0,
                rel_tolerance=0.005,
            ),
            MetricDefinition(
                metric_id="gross_revenue",
                metric_name="Gross Revenue",
                description="Total positive sales revenue excluding returns and cancellations.",
                required_columns=["Quantity", "UnitPrice", "InvoiceNo", "InvoiceDate"],
                formula="SUM(Quantity * UnitPrice) for non-cancelled rows with Quantity > 0",
                supported_operations=["value", "growth_pct", "share_pct"],
                applicable_filters=["period", "product", "country"],
                unit="currency",
                validation_rules=["Exclude 'C' invoice prefixes and negative quantities"],
                abs_tolerance=1.0,
                rel_tolerance=0.005,
            ),
            MetricDefinition(
                metric_id="returns_amount",
                metric_name="Returns Amount",
                description="Absolute or signed revenue impact of negative-quantity return rows.",
                required_columns=["Quantity", "UnitPrice", "InvoiceNo", "InvoiceDate"],
                formula="ABS(SUM(Quantity * UnitPrice)) for return transactions (Quantity < 0, non-cancellation)",
                supported_operations=["value", "count", "share_pct"],
                applicable_filters=["period", "product", "country"],
                unit="currency",
                validation_rules=["Non-cancellation rows with Quantity < 0"],
                abs_tolerance=1.0,
                rel_tolerance=0.005,
            ),
            MetricDefinition(
                metric_id="net_revenue",
                metric_name="Net Revenue",
                description="Net revenue after accounting for returns and cancellations.",
                required_columns=["Quantity", "UnitPrice", "InvoiceNo", "InvoiceDate"],
                formula="SUM(Quantity * UnitPrice) across sales, returns, and cancellations",
                supported_operations=["value", "growth_pct", "share_pct"],
                applicable_filters=["period", "product", "country"],
                unit="currency",
                validation_rules=["Includes negative quantities and cancellation rows"],
                abs_tolerance=1.0,
                rel_tolerance=0.005,
            ),
            MetricDefinition(
                metric_id="monthly_revenue",
                metric_name="Monthly Revenue",
                description="Revenue aggregated for a specific YYYY-MM month.",
                required_columns=["Quantity", "UnitPrice", "InvoiceDate"],
                formula="SUM(Quantity * UnitPrice) grouped by PeriodMonth",
                supported_operations=["value", "growth_pct", "share_pct", "trend"],
                applicable_filters=["period", "product", "country"],
                unit="currency",
                validation_rules=["Reporting period must resolve to YYYY-MM"],
                abs_tolerance=1.0,
                rel_tolerance=0.005,
            ),
            MetricDefinition(
                metric_id="revenue_growth",
                metric_name="Revenue Growth",
                description="Percentage change in revenue between comparison_period and period.",
                required_columns=["Quantity", "UnitPrice", "InvoiceDate"],
                formula="((Revenue(period) - Revenue(comparison_period)) / ABS(Revenue(comparison_period))) * 100",
                supported_operations=["growth_pct", "value"],
                applicable_filters=["period", "comparison_period", "product", "country"],
                unit="percent",
                validation_rules=["Requires non-zero comparison_period revenue"],
                abs_tolerance=0.25,
                rel_tolerance=0.01,
            ),
            MetricDefinition(
                metric_id="quantity_sold",
                metric_name="Quantity Sold",
                description="Total units sold (net or filtered by product/country/period).",
                required_columns=["Quantity", "InvoiceDate"],
                formula="SUM(Quantity) under active transaction policy",
                supported_operations=["value", "growth_pct", "share_pct", "ranking"],
                applicable_filters=["period", "product", "country"],
                unit="units",
                validation_rules=["Quantity must be numeric"],
                abs_tolerance=1.0,
                rel_tolerance=0.005,
            ),
            MetricDefinition(
                metric_id="product_revenue",
                metric_name="Product Revenue",
                description="Revenue generated by a specific product (Description or StockCode).",
                required_columns=["Description", "Quantity", "UnitPrice", "InvoiceDate"],
                formula="SUM(Quantity * UnitPrice) filtered by product Description/StockCode",
                supported_operations=["value", "growth_pct", "share_pct", "ranking"],
                applicable_filters=["period", "product", "country"],
                unit="currency",
                validation_rules=["Product entity must match dataset catalog"],
                abs_tolerance=1.0,
                rel_tolerance=0.005,
            ),
            MetricDefinition(
                metric_id="top_products",
                metric_name="Top-Performing Products",
                description="Ranking of products by revenue.",
                required_columns=["Description", "Quantity", "UnitPrice"],
                formula="ORDER BY SUM(Quantity * UnitPrice) DESC per Description",
                supported_operations=["ranking", "value"],
                applicable_filters=["period", "country"],
                unit="rank",
                validation_rules=["At least one valid product record"],
                abs_tolerance=0.0,
                rel_tolerance=0.0,
            ),
            MetricDefinition(
                metric_id="country_revenue",
                metric_name="Country Revenue",
                description="Revenue generated in a specific country/market.",
                required_columns=["Country", "Quantity", "UnitPrice", "InvoiceDate"],
                formula="SUM(Quantity * UnitPrice) filtered by Country",
                supported_operations=["value", "growth_pct", "share_pct", "ranking"],
                applicable_filters=["period", "country", "product"],
                unit="currency",
                validation_rules=["Country entity must match dataset"],
                abs_tolerance=1.0,
                rel_tolerance=0.005,
            ),
            MetricDefinition(
                metric_id="average_order_value",
                metric_name="Average Order Value",
                description="Mean revenue per unique invoice/order.",
                required_columns=["InvoiceNo", "Quantity", "UnitPrice"],
                formula="SUM(Quantity * UnitPrice) / COUNT(DISTINCT InvoiceNo)",
                supported_operations=["value", "growth_pct"],
                applicable_filters=["period", "country"],
                unit="currency",
                validation_rules=["Distinct InvoiceNo count must be > 0"],
                abs_tolerance=1.0,
                rel_tolerance=0.01,
            ),
            MetricDefinition(
                metric_id="revenue_contribution",
                metric_name="Revenue Contribution",
                description="Percentage share of total revenue attributable to an entity or period.",
                required_columns=["Quantity", "UnitPrice"],
                formula="(Entity Revenue / Total Revenue) * 100",
                supported_operations=["share_pct", "value"],
                applicable_filters=["period", "product", "country"],
                unit="percent",
                validation_rules=["Total denominator revenue must be non-zero"],
                abs_tolerance=0.25,
                rel_tolerance=0.01,
            ),
            MetricDefinition(
                metric_id="cancellation_analysis",
                metric_name="Cancellation Analysis",
                description="Count, value, or rate of cancelled invoices ('C' prefix).",
                required_columns=["InvoiceNo", "Quantity", "UnitPrice"],
                formula="COUNT / SUM(ABS(Quantity * UnitPrice)) / Rate for InvoiceNo starting with 'C'",
                supported_operations=["value", "count", "share_pct"],
                applicable_filters=["period", "product", "country"],
                unit="currency",
                validation_rules=["Cancellation invoices identified by 'C' prefix"],
                abs_tolerance=1.0,
                rel_tolerance=0.005,
            ),
            MetricDefinition(
                metric_id="order_count",
                metric_name="Order Count",
                description="Number of distinct invoices/orders.",
                required_columns=["InvoiceNo"],
                formula="COUNT(DISTINCT InvoiceNo)",
                supported_operations=["count", "value"],
                applicable_filters=["period", "country", "product"],
                unit="count",
                validation_rules=["InvoiceNo must be present"],
                abs_tolerance=0.0,
                rel_tolerance=0.0,
            ),
        ]

    def get(self, metric_id: str) -> MetricDefinition | None:
        return self._metrics.get(metric_id)

    def list_all(self) -> list[MetricDefinition]:
        return list(self._metrics.values())

    def is_supported(self, metric_id: str | None) -> bool:
        return bool(metric_id and metric_id in self._metrics)

    @staticmethod
    def is_explicitly_unsupported_phrase(text: str | None) -> bool:
        if not text:
            return False
        lowered = text.lower()
        return any(kw in lowered for kw in UNSUPPORTED_METRIC_KEYWORDS)
