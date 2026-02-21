schema = {
  "table": "nielsen_pos",
  "columns": {
    "temporal": {
      "period_date": """
        - type: DATE
        - period_date represents data with weekly granularity, specifically on Saturdays, in the format (yyyy-mm-dd).
        - ALWAYS use the period_date column for weekly aggregations or when any kind of filter requires incomplete months.
        - Be careful when filtering this column, as data exists only for Saturdays.
        - For higher time granularity, use the 'year_month' or 'quarter' columns.
      """,
      "year_month": """
        - type: DATE
        - ALWAYS use the year_month column for monthly aggregations and filters, as all date values within a month are mapped to the first day of that month (yyyy-mm-01).
        - Any queries or filters that require complete month or multiple complete months' data, should utilize the year_month column to ensure accurate and consistent results.
        - Because the year_month column already represents the month itself not transformations are needed.
      """,
      "quarter_nielsen": """
        - type: STRING
        - quarter_nielsen represents the quarter of the year and is extracted from the year_month column. Values = ['Q1', 'Q2', 'Q3', 'Q4']
      """,
      "year_nielsen": """
        - type: INT
        - year_nielsen represents the year and is extracted from the year_month column.
      """
    },
    "geographic": {
      "market": """
        - type: STRING
        - market represents the market and is extracted from the market column.
        - Any query requiring the market data, should utilize the market column to ensure accurate and consistent results.
        - PATTERN RECOGNITION: Market values typically end with 'SMM xAOC' (e.g., 'New York SMM xAOC', 'Rem US South Atlantic SMM xAOC').
        - FILTER LOGIC: Use market filter ONLY if no division or customer is specified. Default: 'Total US xAOC + Conv'.
      """,
      "total": """
        - type: STRING
        - total represents the total value and is extracted from the total column.
        - Total = ['Total US xAOC + Conv'] will be used if no customer, or no market or no division is mentioned.
        - Any query requiring the total data, should utilize the total column to ensure accurate and consistent results.
      """,
      "customer": """
        - type: STRING
        - customer represents the retail customer/channel where the product is sold. This can include customer market, retailers, or customer channels (e.g., 'Walmart', 'Target', 'Amazon', etc.)
        - PATTERN RECOGNITION: Customer values typically end with 'TA' (e.g., 'Walmart Total US TA', 'Ahold Delhaize Corp Total TA', 'Dol Gen Total TA').
        - FILTER LOGIC: When customer is specified, use ONLY customer filter (do NOT add market filter).
      """,
      "division": """
        - type: STRING
        - division represents the business division or organizational segment that the product belongs to (e.g., 'Mountain Division xAOC', 'Pacific Division xAOC').
        - PATTERN RECOGNITION: Division values always end with 'Division xAOC' (e.g., 'South Atlantic Division xAOC', 'New England Division xAOC').
        - FILTER LOGIC: When division is specified, use ONLY division filter (do NOT add market filter).
      """
    },
    "category_hierarchy": {
      "mega_category": """
        - type: STRING
        - mega_category represents a higher-level classification of the product category.
      """,
      "category": """
        - type: STRING
        - category represents the primary classification of the product.
      """,
      "sub_category": """
        - type: STRING
        - sub_category represents a more detailed classification within the primary category.
      """
    },
    "product_hierarchy": {
      "manufacturer": """
        - type: STRING
        - manufacturer represents the company or entity that produces the product.
        - Our client is called 'MONDELEZ' (manufacturer), anything else is a competitor.
      """,
      "brand": """
        - type: STRING
        - brand represents the brand name under which the product is marketed.
      """,
      "subbrand": """
        - type: STRING
        - subbrand represents a subcategory within the brand.
      """,
      "ppg": """
        - type: STRING
        - ppg represents an aggregation of multiple similar products (also known as Promoted Package Groups or Promoted Product Groups or Product Groups).
      """,
      "product_name": """
        - type: STRING
        - product_name represents the name of the product.
      """,
      "upc": """
        - type: INT
        - upc (Universal Product Code) represents the unique identifier for the product.
      """,
      "pack": """
        - type: STRING
        - pack represents the packaging configuration of the product.
      """,
      "pack_type": """
        - type: STRING
        - pack_type represents the type of packaging, 'SINGLE', 'MULTI-PACK'.
      """
    },
    "base_metrics": {
      "sales_dollar": """
        - type: FLOAT
        - sales_dollar represents sales in dollars. Use this column when querying for sales revenue in dollar amounts. Default column for sales.
      """,
      "sales_units": """
        - type: FLOAT
        - sales_units represents sales in units. Use this column for querying sales units/volume.
      """,
      "sales_lbs": """
        - type: FLOAT
        - sales_lbs represents sales in weighted units (pounds).
      """,
      "tdp": """
        - type: FLOAT
        - tdp (Total Distribution Points) represents the number of total distribution points. Its unit for TDP is absolute number but, for TDP Growth its unit is percentage (%).
        - At the product/item/upc level, TDP (Total Distribution Points) is equal to % ACV (All Commodity Volume).
      """,
      "display": """
        - type: FLOAT
        - display represents the number of displays in-store, indicating product visibility.
        - display cannot be aggregated by "SUM" since multiple products can occupy the same display.
        - Instead, display should be aggregated using a weighted average.
      """
    },
    "promotional_dollar": {
      "sales_dollar_without_promo": """
        - type: FLOAT
        - sales_dollar_without_promo represents sales in dollars without any promotion (no price reduction and no feature/display support).
      """,
      "sales_dollar_with_any_promo": """
        - type: FLOAT
        - sales_dollar_with_any_promo represents sales in dollars with any promotion (price reduction and/or feature/display support).
      """,
      "incremental_sales_dollar_due_to_promo": """
        - type: FLOAT
        - incremental_sales_dollar_due_to_promo represents the additional sales in dollars attributed to promotional activities.
      """,
      "sales_dollar_with_feature_and_display": """
        - type: FLOAT
        - sales_dollar_with_feature_and_display represents sales in dollars with both feature and display promotions.
      """,
      "sales_dollar_with_feature": """
        - type: FLOAT
        - sales_dollar_with_feature represents sales in dollars with only feature support.
      """,
      "sales_dollar_with_feature_or_display": """
        - type: FLOAT
        - sales_dollar_with_feature_or_display represents sales in dollars with either feature or display promotions.
      """,
      "sales_dollar_with_only_feature": """
        - type: FLOAT
        - sales_dollar_with_only_feature represents sales in dollars with only feature promotions, without any other promotional support.
      """,
      "sales_dollar_with_only_display": """
        - type: FLOAT
        - sales_dollar_with_only_display represents sales in dollars with only display promotions, without any other promotional support.
      """,
      "sales_dollar_with_price_decrease": """
        - type: FLOAT
        - sales_dollar_with_price_decrease represents sales in dollars with a temporary price reduction (discounts), but without feature or display support.
      """
    },
    "promotional_units": {
      "sales_units_without_promo": """
        - type: FLOAT
        - sales_units_without_promo represents sales in units without any promotion (no price reduction and no feature/display support).
      """,
      "sales_units_with_any_promo": """
        - type: FLOAT
        - sales_units_with_any_promo represents sales in units with any promotion (price reduction and/or feature/display support).
      """,
      "incremental_sales_units_due_to_promo": """
        - type: FLOAT
        - incremental_sales_units_due_to_promo represents the additional sales in units attributed to promotional activities.
      """,
      "sales_units_with_feature_and_display": """
        - type: FLOAT
        - sales_units_with_feature_and_display represents sales in units with both feature and display promotions.
      """,
      "sales_units_with_feature": """
        - type: FLOAT
        - sales_units_with_feature represents sales in units with only feature support.
      """,
      "sales_units_with_display": """
        - type: FLOAT
        - sales_units_with_display represents sales in units with only display support.
      """,
      "sales_units_with_feature_or_display": """
        - type: FLOAT
        - sales_units_with_feature_or_display represents sales in units with either feature or display promotions.
      """,
      "sales_units_with_only_feature": """
        - type: FLOAT
        - sales_units_with_only_feature represents sales in units with only feature promotions, without any other promotional support.
      """,
      "sales_units_with_only_display": """
        - type: FLOAT
        - sales_units_with_only_display represents sales in units with only display promotions, without any other promotional support.
      """,
      "sales_units_with_price_decrease": """
        - type: FLOAT
        - sales_units_with_price_decrease represents sales in units with a temporary price reduction (discounts), but without feature or display support.
      """
    },
    "promotional_tdp": {
      "tdp_with_any_promo": """
        - type: FLOAT
        - tdp_with_any_promo represents the total distribution points with any promotion (price reduction and/or feature/display support).
      """,
      "tdp_with_feature_and_display": """
        - type: FLOAT
        - tdp_with_feature_and_display represents total distribution points for products promoted with both feature and display.
      """,
      "tdp_with_display": """
        - type: FLOAT
        - tdp_with_display represents total distribution points for products promoted with only display support.
      """,
      "tdp_with_feature_or_display": """
        - type: FLOAT
        - tdp_with_feature_or_display represents total distribution points for products promoted with either feature or display.
      """,
      "tdp_with_only_feature": """
        - type: FLOAT
        - tdp_with_only_feature represents total distribution points for products promoted with only feature support, without any other promotional support.
      """,
      "tdp_with_only_display": """
        - type: FLOAT
        - tdp_with_only_display represents total distribution points for products promoted with only display support, without any other promotional support.
      """,
      "tdp_with_price_decrease": """
        - type: FLOAT
        - tdp_with_price_decrease represents total distribution points for products promoted with a price reduction, without feature or display support.
      """
    }
  }
}
