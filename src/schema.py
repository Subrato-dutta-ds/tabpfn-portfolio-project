FEATURE_SCHEMA = {
    "numerical": {
        "age": {"min": 17, "max": 100},
        "campaign": {"min": 1, "max": 100},
        "pdays": {"min": 0, "max": 999},
        "previous": {"min": 0, "max": 100},
        "emp.var.rate": {"min": -5.0, "max": 5.0},
        "cons.price.idx": {"min": 90.0, "max": 100.0},
        "cons.conf.idx": {"min": -60.0, "max": 0.0},
        "euribor3m": {"min": 0.0, "max": 10.0},
        "nr.employed": {"min": 4900.0, "max": 5300.0},
    },
    "categorical": {
        "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"],
        "marital": ["divorced", "married", "single", "unknown"],
        "education": ["basic.4y", "basic.6y", "basic.9y", "high.school",
                      "illiterate", "professional.course", "university.degree", "unknown"],
        "default": ["no", "unknown", "yes"],
        "housing": ["no", "unknown", "yes"],
        "loan": ["no", "unknown", "yes"],
        "contact": ["cellular", "telephone"],
        "month": ["apr", "aug", "dec", "jul", "jun", "mar", "may", "nov", "oct", "sep"],
        "day_of_week": ["fri", "mon", "thu", "tue", "wed"],
        "poutcome": ["failure", "nonexistent", "success"],
    }
}

# API aliases: mapping from Python-friendly names to dataset column names
API_ALIASES = {
    "emp_var_rate": "emp.var.rate",
    "cons_price_idx": "cons.price.idx",
    "cons_conf_idx": "cons.conf.idx",
    "nr_employed": "nr.employed",
}

# Reverse: dataset column -> API name
DATASET_TO_API = {v: k for k, v in API_ALIASES.items()}

TARGET_COLUMN = "y"
DROPPED_COLUMNS = ["duration"]
RANDOM_STATE = 42
REVENUE_PER_SUBSCRIPTION = 2000
COST_PER_CONTACT = 50
