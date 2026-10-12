import sqlite3
from datetime import datetime


DATABASE = "attendance.db"

employee_id = "6486"


today = datetime.now().strftime("%Y-%m-%d")


connection = sqlite3.connect(DATABASE)

cursor = connection.cursor()


cursor.execute("""
    DELETE FROM attendance
    WHERE employee_id = ?
    AND date = ?
""", (
    employee_id,
    today
))


deleted_rows = cursor.rowcount


connection.commit()

connection.close()


if deleted_rows > 0:

    print(
        f"Today's attendance for {employee_id} "
        "was removed successfully."
    )

else:

    print(
        f"No attendance record found for "
        f"{employee_id} today."
    )       