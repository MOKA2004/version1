from email_service import send_late_email


send_late_email(
    employee_name="Employee 6495",
    employee_email="YOUR_TEST_EMAIL@gmail.com",
    check_in="19:50",
    shift_start="14:00",
    late_minutes=350
)