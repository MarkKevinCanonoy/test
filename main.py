from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import mysql.connector
from mysql.connector import Error
import bcrypt
import jwt
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# --- configuration setup ---

# 1. load the variables from .env file
load_dotenv()

# 2. get keys safely
API_KEY = os.getenv("GOOGLE_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")

# 3. configure google ai
if not API_KEY:
    print("warning: google_api_key not found in .env file")
else:
    genai.configure(api_key=API_KEY)
    # using 'lite' model for speed
    model = genai.GenerativeModel('gemma-3-12b-it')

# 4. database config
DB_CONFIG = {
    'host': os.getenv("DB_HOST", "localhost"),
    'user': os.getenv("DB_USER", "root"),
    'password': os.getenv("DB_PASSWORD", "kurt_cobain"),
    'port': int(os.getenv("DB_PORT", 3306)),
    'database': os.getenv("DB_NAME", "school_clinic")
}

# --- OPTIMIZATION: SMARTER AI INSTRUCTIONS ---
# REWRITTEN to force the AI to ask questions if data is missing.

BASE_INSTRUCTION = """
ROLE: You are a smart, empathetic School Clinic Receptionist. 

YOUR GOAL: Collect 4 pieces of information to book an appointment:
1. **Service Type** (Must be "Medical Consultation" or "Medical Clearance")
2. **Date** (YYYY-MM-DD)
3. **Time** (24-hour format)
4. **Reason** (Short description)

YOUR RULES:

1. **MISSING INFO CHECK (Crucial):**
   - If the user says "Book appointment" but didn't say what for, ASK: "Is this for a Medical Consultation or Medical Clearance?"
   - If the user didn't say a date/time, ASK: "When would you like to schedule that?"
   - If the user didn't give a reason, ASK: "What is the reason for your visit?"
   - **DO NOT** output the JSON booking action until you have ALL 4 pieces of info.

2. **ONE-LINER HANDLING:**
   - If the user provides EVERYTHING at once (e.g., "I need a clearance tomorrow at 2pm"), then you can skip asking and output the JSON immediately.

3. **SMART EXTRACTION:**
   - **Service:** If they say "checkup" or "doc", assume "Medical Consultation". If they say "pass" or "clearance", assume "Medical Clearance".
   - **Time:** Convert "2pm", "2 in the afternoon" -> "14:00:00". Convert "10am" -> "10:00:00".

4. **MEDICAL ADVICE:** - If the user mentions pain (headache, fever), suggest a simple remedy (water, rest) *while* you ask for the booking details.

5. **CANCELLATION:**
   - If the user says "Cancel #5" or "Remove ID 5", output the Cancel JSON.

🔴 OUTPUT FORMAT (JSON ONLY for Actions):

[ACTION 1: BOOKING - Only output when ALL 4 fields are ready] 
{
  "action": "book_appointment",
  "date": "YYYY-MM-DD",
  "time": "HH:MM:00",
  "reason": "extracted reason",
  "service_type": "Medical Consultation" OR "Medical Clearance", 
  "urgency": "Normal" (or "Urgent" if pain is severe),
  "ai_advice": "Drink water." (Optional)
}

[ACTION 2: CANCELING]
{
  "action": "cancel_appointment",
  "appointment_id": 123
}
"""

# --- helper functions ---

def get_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        raise HTTPException(status_code=500, detail=f"database connection failed: {str(e)}")

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# jwt configuration
ALGORITHM = "HS256"
security = HTTPBearer()

def create_token(user_id: int, role: str, full_name: str) -> str:
    payload = {
        'user_id': user_id,
        'role': role,
        'full_name': full_name,
        'exp': datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    return decode_token(token)

def create_default_users():
    conn = get_db()
    cursor = conn.cursor()
    try:
        default_users = [
            {"full_name": "Super Admin", "email": "superadmin@clinic.com", "password": "admin123", "role": "super_admin"},
            {"full_name": "Clinic Admin", "email": "admin@clinic.com", "password": "admin123", "role": "admin"}
        ]
        for user in default_users:
            cursor.execute("SELECT id FROM users WHERE email = %s", (user['email'],))
            if not cursor.fetchone():
                hashed_pw = hash_password(user['password'])
                cursor.execute(
                    "INSERT INTO users (full_name, email, password, role) VALUES (%s, %s, %s, %s)",
                    (user['full_name'], user['email'], hashed_pw, user['role'])
                )
                print(f"created default user: {user['email']}")
        conn.commit()
    except Error as e:
        print(f"error seeding database: {e}")
    finally:
        cursor.close()
        conn.close()

# --- main app setup ---
app = FastAPI()

@app.on_event("startup")
def on_startup():
    create_default_users()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- pydantic models ---
class UserRegister(BaseModel):
    full_name: str
    email: EmailStr
    password: str

class AdminCreateUser(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    role: str 

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class AppointmentCreate(BaseModel):
    appointment_date: str
    appointment_time: str
    service_type: str
    urgency: str
    reason: str
    booking_mode: str = "standard"

class AppointmentUpdate(BaseModel):
    status: str
    admin_note: Optional[str] = None

class ChatMessage(BaseModel):
    message: str
    history: List[dict] = []

# --- api routes ---

@app.post("/api/register")
def register(user: UserRegister):
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE email = %s", (user.email,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="email already registered")
        
        hashed_pw = hash_password(user.password)
        cursor.execute(
            "INSERT INTO users (full_name, email, password, role) VALUES (%s, %s, %s, 'student')",
            (user.full_name, user.email, hashed_pw)
        )
        conn.commit()
        return {"message": "registration successful"}
    except Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.post("/api/admin/create-user")
def create_admin_user(user: AdminCreateUser, current_user = Depends(get_current_user)):
    if current_user['role'] != 'super_admin':
        raise HTTPException(status_code=403, detail="only super admins can create admin accounts")
    
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE email = %s", (user.email,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="email already registered")
        
        hashed_pw = hash_password(user.password)
        cursor.execute(
            "INSERT INTO users (full_name, email, password, role) VALUES (%s, %s, %s, %s)",
            (user.full_name, user.email, hashed_pw, user.role)
        )
        conn.commit()
        return {"message": f"user created successfully as {user.role}"}
    finally:
        cursor.close()
        conn.close()

@app.post("/api/login")
def login(user: UserLogin):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM users WHERE email = %s", (user.email,))
        db_user = cursor.fetchone()
        
        if not db_user or not verify_password(user.password, db_user['password']):
            raise HTTPException(status_code=401, detail="invalid email or password")
        
        token = create_token(db_user['id'], db_user['role'], db_user['full_name'])
        
        return {
            "token": token,
            "role": db_user['role'],
            "user_id": db_user['id'],
            "full_name": db_user['full_name']
        }
    finally:
        cursor.close()
        conn.close()

@app.get("/api/appointments")
def get_appointments(current_user = Depends(get_current_user)):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    try:
        if current_user['role'] == 'student':
            cursor.execute("""
                SELECT a.*, u.full_name as student_name
                FROM appointments a
                JOIN users u ON a.student_id = u.id
                WHERE a.student_id = %s
                ORDER BY a.appointment_date DESC, a.appointment_time DESC
            """, (current_user['user_id'],))
        else:
            cursor.execute("""
                SELECT a.*, u.full_name as student_name, u.email as student_email
                FROM appointments a
                JOIN users u ON a.student_id = u.id
                ORDER BY a.appointment_date DESC, a.appointment_time DESC
            """)
        
        results = cursor.fetchall()
        for row in results:
            row['appointment_date'] = str(row['appointment_date'])
            row['appointment_time'] = str(row['appointment_time'])
            
        return results
    finally:
        cursor.close()
        conn.close()

@app.post("/api/appointments")
def create_appointment(appointment: AppointmentCreate, current_user = Depends(get_current_user)):
    if current_user['role'] != 'student':
        raise HTTPException(status_code=403, detail="only students can book appointments")
    
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO appointments (student_id, appointment_date, appointment_time, service_type, urgency, reason, booking_mode, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending')
        """, (
            current_user['user_id'], 
            appointment.appointment_date, 
            appointment.appointment_time, 
            appointment.service_type, 
            appointment.urgency, 
            appointment.reason, 
            appointment.booking_mode
        ))
        conn.commit()
        return {"message": "appointment booked successfully", "id": cursor.lastrowid}
    except Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.put("/api/appointments/{appointment_id}")
def update_appointment(appointment_id: int, update: AppointmentUpdate, current_user = Depends(get_current_user)):
    if current_user['role'] not in ['admin', 'super_admin']:
        raise HTTPException(status_code=403, detail="only admins can update appointments")
    
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT status FROM appointments WHERE id = %s", (appointment_id,))
        current_appt = cursor.fetchone()
        
        if not current_appt:
            raise HTTPException(status_code=404, detail="appointment not found")

        if update.status == 'completed' and current_appt['status'] == 'completed':
            raise HTTPException(status_code=400, detail="already_scanned")

        cursor.execute("""
            UPDATE appointments 
            SET status = %s, admin_note = %s, updated_at = NOW()
            WHERE id = %s
        """, (update.status, update.admin_note, appointment_id))
        conn.commit()
        
        return {"message": "appointment updated successfully"}
    finally:
        cursor.close()
        conn.close()

@app.delete("/api/appointments/{appointment_id}")
def delete_or_cancel_appointment(appointment_id: int, current_user = Depends(get_current_user)):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT student_id, status FROM appointments WHERE id = %s", (appointment_id,))
        appt = cursor.fetchone()
        
        if not appt:
            raise HTTPException(status_code=404, detail="appointment not found")

        if current_user['role'] in ['admin', 'super_admin']:
             cursor.execute("DELETE FROM appointments WHERE id = %s", (appointment_id,))
             message = "appointment permanently deleted."
        elif current_user['role'] == 'student':
            if appt['student_id'] != current_user['user_id']:
                raise HTTPException(status_code=403, detail="not authorized")

            if appt['status'] == 'pending':
                 cursor.execute("UPDATE appointments SET status = 'canceled', updated_at = NOW() WHERE id = %s", (appointment_id,))
                 message = "appointment canceled successfully"
            else:
                 cursor.execute("DELETE FROM appointments WHERE id = %s", (appointment_id,))
                 message = "appointment record deleted successfully"
        else:
             raise HTTPException(status_code=403, detail="action not allowed")
        
        conn.commit()
        return {"message": message}
    finally:
        cursor.close()
        conn.close()

@app.get("/api/users")
def get_users(current_user = Depends(get_current_user)):
    if current_user['role'] != 'super_admin':
        raise HTTPException(status_code=403, detail="only super admins can view users")
    
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, full_name, email, role, created_at FROM users ORDER BY created_at DESC")
        results = cursor.fetchall()
        for row in results:
            row['created_at'] = str(row['created_at'])
        return results
    finally:
        cursor.close()
        conn.close()

@app.delete("/api/users/{user_id}")
def delete_user(user_id: int, current_user = Depends(get_current_user)):
    if current_user['role'] != 'super_admin':
        raise HTTPException(status_code=403, detail="only super admins can delete users")
    if current_user['user_id'] == user_id:
        raise HTTPException(status_code=400, detail="you cannot delete your own account")

    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        return {"message": "user deleted successfully"}
    except Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

# ==========================================
#  SMART AI CHATBOT V2
# ==========================================

@app.post("/api/chat")
async def chat_booking(chat: ChatMessage, current_user = Depends(get_current_user)):
    """
    Enhanced AI Chatbot:
    - Supports One-Liner Booking ("Book checkup tomorrow 2pm")
    - Smart Cancellation ("Cancel #5")
    - Medical Advice ("My head hurts" -> Advice)
    - [NEW] Smart Missing Info Detection
    """
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # step 1: get appointments context
    try:
        cursor.execute("""
            SELECT id, appointment_date, appointment_time, reason 
            FROM appointments 
            WHERE student_id = %s AND status IN ('pending', 'approved')
            ORDER BY appointment_date ASC
        """, (current_user['user_id'],))
        active_appts = cursor.fetchall()
        
        appt_list_text = ""
        if active_appts:
            for appt in active_appts:
                appt_list_text += f"- ID {appt['id']}: {appt['appointment_date']} at {appt['appointment_time']} (Reason: {appt['reason']})\n"
        else:
            appt_list_text = "None."
            
    finally:
        cursor.close()
        conn.close()

    # step 2: dynamic prompt
    final_instruction = f"""
    {BASE_INSTRUCTION}
    
    Student Name: {current_user['full_name']}
    Current Date: {datetime.now().strftime("%Y-%m-%d")}

    CONTEXT (Use this to help the student cancel or reschedule):
    {appt_list_text}
    """

    try:
        # step 3: build history
        history_for_google = [
            {"role": "user", "parts": [final_instruction]},
            {"role": "model", "parts": ["Understood. I will check for all 4 booking details before acting. 😊"]}
        ]

        if chat.history:
            recent_msgs = chat.history[-10:] 
            for msg in recent_msgs:
                role = "user" if msg.get("role") == "user" else "model"
                history_for_google.append({
                    "role": role,
                    "parts": [msg.get("message", "")]
                })

        # step 4: generate response
        chat_session = model.start_chat(history=history_for_google)
        response = chat_session.send_message(chat.message)
        ai_text = response.text

        # step 5: check for json actions
        if "{" in ai_text and "}" in ai_text:
            try:
                start = ai_text.find('{')
                end = ai_text.rfind('}') + 1
                data = json.loads(ai_text[start:end])

                # --- ACTION A: BOOKING ---
                if data.get("action") == "book_appointment":
                    conn = get_db()
                    cursor = conn.cursor(buffered=True)
                    
                    # duplicate check
                    cursor.execute("""
                        SELECT id FROM appointments 
                        WHERE appointment_date = %s AND appointment_time = %s AND status != 'canceled'
                    """, (data['date'], data['time']))
                    
                    if cursor.fetchone():
                        cursor.close()
                        conn.close()
                        return {"response": f"⚠️ That time ({data['time']}) is already taken! Please choose another time.", "requires_action": False}

                    # insert
                    cursor.execute("""
                        INSERT INTO appointments (student_id, appointment_date, appointment_time, service_type, urgency, reason, booking_mode, status)
                        VALUES (%s, %s, %s, %s, %s, %s, 'ai_chatbot', 'pending')
                    """, (current_user['user_id'], data['date'], data['time'], data['service_type'], data['urgency'], data['reason']))
                    
                    conn.commit()
                    cursor.close()
                    conn.close()
                    
                    # success message + advice if available
                    success_msg = f"Booked for {data['date']} at {data['time']}! ✅"
                    if data.get("ai_advice"):
                        success_msg += f"\n\n💡 Health Tip: {data['ai_advice']}"
                        
                    return {"response": success_msg, "requires_action": False}

                # --- ACTION B: CANCELING ---
                elif data.get("action") == "cancel_appointment":
                    appt_id = data.get("appointment_id")
                    
                    conn = get_db()
                    cursor = conn.cursor()
                    
                    # verify
                    cursor.execute("SELECT id FROM appointments WHERE id = %s AND student_id = %s", (appt_id, current_user['user_id']))
                    
                    if cursor.fetchone():
                        cursor.execute("UPDATE appointments SET status = 'canceled' WHERE id = %s", (appt_id,))
                        conn.commit()
                        msg = f"Okay, appointment #{appt_id} has been canceled. 🗑️"
                    else:
                        msg = f"I couldn't find Appointment #{appt_id}. Please check the list."
                        
                    cursor.close()
                    conn.close()
                    return {"response": msg, "requires_action": False}

            except Exception as e:
                print(f"json processing error: {e}")
                return {"response": "System error processing your request. Please try again.", "requires_action": False}

        # normal reply
        return {"response": ai_text, "requires_action": False}

    except Exception as e:
        error_msg = str(e)
        print(f"ai error: {error_msg}")
        return {"response": "My AI brain is a bit busy. Please try again in 10 seconds.", "requires_action": False}



# ADD THIS AT THE BOTTOM (To serve your HTML files)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Mount the current directory to serve HTML/CSS/JS
if os.path.exists("images"):
    app.mount("/images", StaticFiles(directory="images"), name="images")

# 2. Serve CSS files explicitly
@app.get("/style.css")
async def get_style():
    return FileResponse("style.css")

@app.get("/dashboard.css")
async def get_dash_css():
    return FileResponse("dashboard.css")

# 3. Serve JavaScript files explicitly
@app.get("/admin-dashboard.js")
async def get_admin_js():
    return FileResponse("admin-dashboard.js")

@app.get("/student-dashboard.js")
async def get_student_js():
    return FileResponse("student-dashboard.js")

@app.get("/login.js")
async def get_login_js():
    return FileResponse("login.js")

@app.get("/register.js")
async def get_reg_js():
    return FileResponse("register.js")

@app.get("/main.js")
async def get_main_js():
    return FileResponse("main.js")

# 4. Serve the HTML Pages
@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.get("/register")
async def read_register():
    return FileResponse('register.html')

@app.get("/student-dashboard")
async def read_student():
    return FileResponse('student-dashboard.html')

@app.get("/admin-dashboard")
async def read_admin():
    return FileResponse('admin-dashboard.html')


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)