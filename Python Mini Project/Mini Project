

from __future__ import annotations

import datetime as dt
import os
import sqlite3
from collections import defaultdict

import numpy as np
from flask import Flask, flash, g, redirect, render_template, request, url_for
from sklearn.ensemble import RandomForestClassifier



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "pantry.db")

app = Flask(__name__)
app.secret_key = "food-waste-prediction-dev-key"

CATEGORIES = {
    "Dairy":     0.50,
    "Bakery":    0.50,
    "Produce":   0.60,
    "Meat":      0.30,
    "Grains":    0.15,
    "Frozen":    0.20,
    "Canned":    0.10,
    "Beverages": 0.50,
    "Snacks":    0.30,
}
CAT_IDS = {c: i for i, c in enumerate(CATEGORIES)}



def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    db = sqlite3.connect(DB_PATH)
    legacy = {}
    for table, cols in (("items", "id,name,category,qty,unit,added,expiry"),
                        ("log", "id,name,category,qty,action,date")):
        existing = [r[1] for r in db.execute(f"PRAGMA table_info({table})")]
        if "price" in existing:
            db.execute(f"ALTER TABLE {table} RENAME TO {table}_old")
            legacy[table] = cols

    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            qty REAL NOT NULL,
            unit TEXT NOT NULL DEFAULT 'pcs',
            added TEXT NOT NULL,
            expiry TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            qty REAL NOT NULL,
            action TEXT NOT NULL,          -- 'used' or 'wasted'
            date TEXT NOT NULL
        );
        """
    )
    for table, cols in legacy.items():
        db.execute(f"INSERT INTO {table}({cols}) SELECT {cols} FROM {table}_old")
        db.execute(f"DROP TABLE {table}_old")
    db.commit()
    db.close()


def today() -> dt.date:
    return dt.date.today()



def learned_rates(db: sqlite3.Connection) -> dict[str, float]:
    """Blend default category consumption rates with the user's real usage."""
    rates = dict(CATEGORIES)
    rows = db.execute(
        "SELECT category, SUM(qty) AS q, MIN(date) AS first FROM log "
        "WHERE action='used' GROUP BY category"
    ).fetchall()
    for r in rows:
        span = max(7, (today() - dt.date.fromisoformat(r["first"])).days + 1)
        observed = r["q"] / span
        weight = min(0.7, span / 60)          
        rates[r["category"]] = (1 - weight) * rates[r["category"]] + weight * observed
    return rates


def _features(cat: str, qty: float, days_left: float, rate: float) -> list[float]:
    expected = max(rate * max(days_left, 0), 0.01)
    return [CAT_IDS[cat], qty, days_left, rate, qty / expected]


_MODEL: RandomForestClassifier | None = None


def get_model() -> RandomForestClassifier:
    """Train once on simulated household behaviour (cached)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    rng = np.random.default_rng(7)
    X, y = [], []
    cats = list(CATEGORIES)
    for _ in range(6000):
        cat = cats[rng.integers(len(cats))]
        rate = CATEGORIES[cat] * rng.uniform(0.5, 1.6)
        qty = float(rng.integers(1, 12))
        days_left = float(rng.integers(-1, 21))
        behaviour = rng.lognormal(0, 0.35)         
        consumed = rate * max(days_left, 0) * behaviour
        wasted = 1 if (qty > consumed or days_left <= 0) else 0
        X.append(_features(cat, qty, days_left, rate))
        y.append(wasted)
    _MODEL = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=7)
    _MODEL.fit(np.array(X), np.array(y))
    return _MODEL


def analyse_pantry(db: sqlite3.Connection) -> list[dict]:
    model = get_model()
    rates = learned_rates(db)
    items = db.execute("SELECT * FROM items ORDER BY expiry").fetchall()
    out = []
    for it in items:
        days_left = (dt.date.fromisoformat(it["expiry"]) - today()).days
        feats = _features(it["category"], it["qty"], days_left, rates[it["category"]])
        risk = float(model.predict_proba([feats])[0][1])
        if days_left < 0:
            risk = 1.0
        level = "high" if risk >= 0.7 else "medium" if risk >= 0.4 else "low"
        out.append({
            **dict(it),
            "days_left": days_left,
            "risk": round(risk * 100),
            "level": level,
        })
    return out


@app.route("/")
def dashboard():
    db = get_db()
    analysed = analyse_pantry(db)
    expiring = [a for a in analysed if 0 <= a["days_left"] <= 3]
    expired = [a for a in analysed if a["days_left"] < 0]
    high_risk = [a for a in analysed if a["level"] == "high"]
    stats = {
        "total": len(analysed),
        "expiring": len(expiring),
        "expired": len(expired),
        "high_risk": len(high_risk),
    }
    return render_template("dashboard.html", items=analysed, stats=stats,
                           alerts=sorted(expired + expiring, key=lambda a: a["days_left"]))


@app.route("/add", methods=["GET", "POST"])
def add_item():
    if request.method == "POST":
        f = request.form
        try:
            name = f["name"].strip()
            qty = float(f["qty"])
            expiry = dt.date.fromisoformat(f["expiry"])
            if not name or qty <= 0:
                raise ValueError
        except (KeyError, ValueError):
            flash("Please fill in a valid name, quantity and expiry date.", "error")
            return redirect(url_for("add_item"))
        db = get_db()
        db.execute(
            "INSERT INTO items(name,category,qty,unit,added,expiry) VALUES(?,?,?,?,?,?)",
            (name, f["category"], qty, f.get("unit", "pcs"),
             today().isoformat(), expiry.isoformat()),
        )
        db.commit()
        flash(f"Added {name} to your pantry.", "success")
        return redirect(url_for("dashboard"))
    default_expiry = (today() + dt.timedelta(days=7)).isoformat()
    return render_template("add.html", categories=list(CATEGORIES), default_expiry=default_expiry)


@app.route("/item/<int:item_id>/<action>", methods=["POST"])
def item_action(item_id: int, action: str):
    db = get_db()
    it = db.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
    if it is None:
        flash("Item not found.", "error")
        return redirect(url_for("dashboard"))
    if action in ("used", "wasted"):
        db.execute(
            "INSERT INTO log(name,category,qty,action,date) VALUES(?,?,?,?,?)",
            (it["name"], it["category"], it["qty"], action, today().isoformat()),
        )
        flash(f"Logged {it['name']} as {action}.", "success")
    elif action != "delete":
        flash("Unknown action.", "error")
        return redirect(url_for("dashboard"))
    db.execute("DELETE FROM items WHERE id=?", (item_id,))
    db.commit()
    return redirect(request.referrer or url_for("dashboard"))


@app.route("/report")
def report():
    db = get_db()
    rows = db.execute("SELECT * FROM log").fetchall()
    wasted = [r for r in rows if r["action"] == "wasted"]
    total_units = sum(r["qty"] for r in rows)
    waste_rate = 100 * sum(r["qty"] for r in wasted) / total_units if total_units else 0

    totals = defaultdict(float)
    wasted_qty = defaultdict(float)
    for r in rows:
        totals[r["category"]] += r["qty"]
        if r["action"] == "wasted":
            wasted_qty[r["category"]] += r["qty"]
    breakdown = sorted(
        ((c, round(100 * wasted_qty[c] / t, 1)) for c, t in totals.items() if t > 0),
        key=lambda kv: kv[1], reverse=True,
    )

    tips = []
    if breakdown and breakdown[0][1] > 0:
        tips.append(f"{breakdown[0][0]} has the highest waste rate ({breakdown[0][1]}%) - "
                    "buy smaller quantities or use it sooner.")
    if waste_rate > 25:
        tips.append("Over a quarter of your food is wasted. Check the expiry alerts on the "
                    "dashboard and use items that are about to expire first.")
    if not tips:
        tips.append("Log items as used or wasted to unlock personalised insights.")

    return render_template(
        "report.html", waste_rate=round(waste_rate, 1), used_count=len(rows) - len(wasted),
        wasted_count=len(wasted), breakdown=breakdown, tips=tips, history=rows[-15:][::-1],
    )


@app.route("/seed", methods=["POST"])
def seed():
    """Load demo data so the app can be demonstrated instantly."""
    db = get_db()
    d = today()
    demo = [
        ("Milk", "Dairy", 2, "L", 1), ("Bread", "Bakery", 1, "loaf", 0),
        ("Chicken Breast", "Meat", 1, "kg", 2), ("Tomato", "Produce", 6, "pcs", 3),
        ("Lettuce", "Produce", 2, "pcs", 2), ("Eggs", "Dairy", 12, "pcs", 10),
        ("Yogurt", "Dairy", 4, "cups", 5), ("Rice", "Grains", 2, "kg", 180),
        ("Pasta", "Grains", 1, "kg", 200), ("Cheese", "Dairy", 1, "block", 12),
        ("Banana", "Produce", 6, "pcs", 2), ("Onion", "Produce", 5, "pcs", 14),
        ("Ice Cream", "Frozen", 1, "tub", 60), ("Orange Juice", "Beverages", 1, "L", -1),
    ]
    for name, cat, qty, unit, days in demo:
        db.execute(
            "INSERT INTO items(name,category,qty,unit,added,expiry) VALUES(?,?,?,?,?,?)",
            (name, cat, qty, unit, d.isoformat(), (d + dt.timedelta(days=days)).isoformat()),
        )
    history = [("Milk", "Dairy", 4, "used", 5), ("Milk", "Dairy", 1, "wasted", 3),
               ("Bread", "Bakery", 3, "used", 6), ("Bread", "Bakery", 2, "wasted", 2),
               ("Spinach", "Produce", 3, "wasted", 4), ("Spinach", "Produce", 1, "used", 9),
               ("Chicken Breast", "Meat", 2, "used", 8), ("Eggs", "Dairy", 12, "used", 7),
               ("Yogurt", "Dairy", 6, "used", 10), ("Banana", "Produce", 6, "wasted", 6)]
    for name, cat, qty, act, ago in history:
        db.execute(
            "INSERT INTO log(name,category,qty,action,date) VALUES(?,?,?,?,?)",
            (name, cat, qty, act, (d - dt.timedelta(days=ago)).isoformat()),
        )
    db.commit()
    flash("Demo data loaded.", "success")
    return redirect(url_for("dashboard"))


@app.route("/reset", methods=["POST"])
def reset():
    db = get_db()
    db.execute("DELETE FROM items")
    db.execute("DELETE FROM log")
    db.commit()
    flash("All data cleared.", "success")
    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    init_db()
    get_model()  
    print("Food Waste Prediction System running at http://127.0.0.1:5000")
    app.run(debug=False)
