import sqlite3

connection = sqlite3.connect("attendance.db")
cursor = connection.cursor()

cursor.execute("""
    SELECT
        employee_id,
        date,
        check_in,
        check_out,
        late_minutes
    FROM attendance
    ORDER BY date DESC
""")

records = cursor.fetchall()

print("\nATTENDANCE RECORDS")
print("------------------")

if not records:
    print("No attendance records found.")

else:
    for record in records:
        employee_id, date, check_in, check_out, late = record

        print(f"Employee   : {employee_id}")
        print(f"Date       : {date}")
        print(f"Check-in   : {check_in}")
        print(f"Check-out  : {check_out}")
        print(f"Late       : {late} minutes")
        print("------------------")

connection.close()