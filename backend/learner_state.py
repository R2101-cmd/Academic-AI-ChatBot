import sqlite3


def init_db():
    conn = sqlite3.connect("learner.db")
    c = conn.cursor()

    c.execute(
        "CREATE TABLE IF NOT EXISTS progress(user TEXT, concept TEXT, score REAL)"
    )

    conn.commit()
    conn.close()


def save(user, concept, score):
    conn = sqlite3.connect("learner.db")
    c = conn.cursor()

    c.execute("INSERT INTO progress VALUES (?, ?, ?)", (user, concept, score))

    conn.commit()
    conn.close()
