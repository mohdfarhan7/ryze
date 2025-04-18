from fastapi import FastAPI, Depends, HTTPException,Body,status ,Form,UploadFile ,File
from pydantic import BaseModel ,ConfigDict
from sqlalchemy import create_engine, Column, Float, Boolean, String, Integer, ForeignKey,DateTime ,Date
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship ,joinedload
from typing import Optional,Dict ,List
import pandas as pd,pickle ,urllib.parse,os ,requests,bcrypt, jwt,psycopg2 ,shutil,json
from sqlalchemy.exc import SQLAlchemyError
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta ,date
from fastapi.staticfiles import StaticFiles
from psycopg2.extras import DictCursor
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# Get database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("❌ DATABASE_URL environment variable is not set.")

# Ensure compatibility with SQLAlchemy
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Media directory configuration
MEDIA_DIR = os.getenv("MEDIA_DIR", "media")
os.makedirs(MEDIA_DIR, exist_ok=True)
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

# Model directory configuration
MODEL_DIR = os.getenv("MODEL_DIR", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Model paths
ISOLATION_FOREST_PATH = os.path.join(MODEL_DIR, "isolation_forest.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Initialize models
try:
    # Load or create isolation forest model
    if os.path.exists(ISOLATION_FOREST_PATH):
        with open(ISOLATION_FOREST_PATH, "rb") as f:
            iso_forest = pickle.load(f)
    else:
        print("Creating new IsolationForest model")
        iso_forest = IsolationForest(random_state=42)
        with open(ISOLATION_FOREST_PATH, "wb") as f:
            pickle.dump(iso_forest, f)

    # Load or create scaler
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    else:
        print("Creating new StandardScaler")
        scaler = StandardScaler()
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

except Exception as e:
    print(f"Warning: Error handling models - {str(e)}")
    print("Initializing new models")
    iso_forest = IsolationForest(random_state=42)
    scaler = StandardScaler()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    user_name = Column(String)  
    email = Column(String, unique=True, nullable=False) 
    password = Column(String, nullable=False)
    mobile = Column(String, nullable=False)
    otp = Column(String, nullable=True)
    otp_expiration = Column(String, nullable=True)
    profile_image = Column(String, nullable=True)
    # Relationships
    expenses = relationship("ExpensesDB", back_populates="user", cascade="all,delete-orphan")
    loan_details = relationship("LoanDetailsDB", back_populates="user", cascade="all, delete-orphan")
    lifestyle = relationship("LifestyleDB", back_populates="user", uselist=False)
    financial_goals = relationship("FinancialGoalsDB", back_populates="user", uselist=False)
    # created & update time
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    #####
class ExpensesDB(Base):
    __tablename__ = "expenses"  
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    income = Column(Float)
    rent = Column(Float)
    groceries = Column(Float)
    transportation = Column(Float)
    healthcare = Column(Float)
    dining_out = Column(Float)
    shopping = Column(Float)
    personal_care = Column(Float)
    education = Column(Float)
    electricity = Column(Float)
    water = Column(Float)
    insurance = Column(Float)
    user = relationship("User", back_populates="expenses")
    #
class LoanDetailsDB(Base):
    __tablename__ = "loan_details"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    loan_exists = Column(Boolean)
    loan_number = Column(String, unique=True, index=True)  
    loan_type = Column(String)
    start_date = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    loan_amount = Column(Float, nullable=True)
    monthly_payment = Column(Float, nullable=True)
    loan_term = Column(Integer, nullable=True)
    interest_rate = Column(Float, nullable=True)
    interest_rate = Column(Float)
    status = Column(String(50), default="active", nullable=True)
    user = relationship("User", back_populates="loan_details")
#
class LifestyleDB(Base):
    __tablename__ = "lifestyle"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    smoke = Column(Boolean)
    dine_out_frequency = Column(String)
    sports_hobbies = Column(String)
    user = relationship("User", back_populates="lifestyle")
#
class FinancialGoalsDB(Base):
    __tablename__ = "financial_goals"    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    goal = Column(String)
    goal_title = Column(String)
    amount = Column(String)
    current_amount =Column(String)
    deadline = Column(String)
    user = relationship("User", back_populates="financial_goals")
#
Base.metadata.create_all(bind=engine) 

class UserCreate(BaseModel):
    user_name: str
    email: str
    password: str
    mobile: str
class LoginRequest(BaseModel):
    email: str
    password: str
class OTPRequest(BaseModel):
    mobile: str
    otp: str        
class ForgotPasswordRequest(BaseModel):
    mobile: str
class OTPVerificationRequest(BaseModel):
    mobile: str
    otp: str
class ResetPasswordRequest(BaseModel):
    id: int
    password: str
class Spending_Request(BaseModel):
    id: int
class UserProfileRequest(BaseModel):
    user_id: int
#    
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
#
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
    return encoded_jwt
@app.post("/Register")
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user.email).first():
        return { "status": "0","message": "User Email already exists","result": {} }
    if db.query(User).filter(User.mobile == user.mobile).first():
        return {"status": "0","message": "Mobile number already registered","result": {}}
    hashed_password = hash_password(user.password)
    db_user = User( user_name=user.user_name,email=user.email,password=hashed_password,mobile=user.mobile )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"status": "1", "message": "User registered successfully!",  "id": db_user.id}
@app.post("/Login")
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    email = request.email.strip()
    user = db.query(User).filter(User.email == email).first() 
    if not user or not verify_password(request.password, user.password):
        return {"status": "0", "message": "Invalid credentials", "result": {}}
    access_token = create_access_token(data={"sub": user.email}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"status": "1","message": "Login successful",
            "result": {"id": user.id,"user_name": user.user_name,"email": user.email,"password": user.password,
        "type": "User","country_code": "","mobile":user.mobile,"image": " ",
        "updated_at": str(user.updated_at) if user.updated_at else "",
        "created_at": str(user.created_at) if user.created_at else "", "device_id": "null",
        "status": "ACTIVE","country": "", "otp": "","city": "","district": "","qr_image": "",
        "qr_code": "","point": "0", "token": access_token}}
def send_otp(mobile: str, otp: str) -> bool:
    print(f"Sending OTP to {mobile}: {otp}")
    return True 
@app.post("/forgot-password")
def forgot_password( request: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.mobile ==request.mobile).first() 
    if not user:
        return {"status": "0", "message": "User not found", "result": {}}
    otp = "9999"
    otp_expiration = (datetime.utcnow() + timedelta(minutes=10)).isoformat()
    user.otp = otp
    user.otp_expiration = otp_expiration
    db.commit()
    if send_otp(user.mobile, otp):
        return {"status": "1", "message": f"OTP sent to {user.mobile}", "result": {"otp": otp} }
    return {"status": "0", "message": "Failed to send OTP", "result": {}}

@app.post("/verify-otp")
def verify_otp(request: OTPVerificationRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.mobile == request.mobile).first() 
    if not user or user.otp != request.otp:
        return {"status": "0", "message": "Invalid mobile number or OTP","results":{}}
    return { "status": "1","message": "Otp Verify",
    "result": {"id": user.id, "user_name": user.user_name, "email": user.email,"password": user.password,
        "type": "User", "country_code": "","mobile":user.mobile,"image": " ", "image": "",
        "updated_at": str(user.updated_at) if user.updated_at else "",
        "created_at": str(user.created_at) if user.created_at else "",
        "device_id": "null","status": "ACTIVE","country": "","otp": "9999",  "city": "","district": "",
        "qr_image": "","qr_code": "","point": "0"}}

@app.post("/update-password")
def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == request.id).first() 
    if not user:
        return {"Status": "0", "Message": "User not found","result":{}}
    user.password = hash_password(request.password)
    db.commit()
    return { "status":"1","message": " Succesfully",
    "user_data": {"id": user.id, "user_name": user.user_name,"email": user.email,"password": user.password,"type": "User",
        "country_code": "","mobile":user.mobile,"image": "",
        "updated_at": str(user.updated_at) if user.updated_at else "",
        "created_at": str(user.created_at) if user.created_at else "",
        "device_id": "null",# "status": "ACTIVE",
        "country": "","otp": "9999","city": "", "district": "","qr_image": "", "qr_code": "",  "point": "0"},"status": "1"}
class DeleteUserRequest(BaseModel):
    user_id: int
@app.delete("/Delete")
def delete_user(data: DeleteUserRequest = Body(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == data.user_id).first()
    if not user:
        return {"status": "0", "message": "User not found"}
    db.delete(user)
    db.commit()
    return {"status": "1", "message": f"User with id {data.user_id} Deleted Successfully"}    
    
class Expenses(BaseModel):
    income: float
    rent: float
    groceries: float
    transportation: float
    healthcare: float
    dining_out: float
    shopping: float
    personal_care: float
    education: float
    electricity: float
    water: float
    insurance: float
    model_config = ConfigDict(from_attributes=True)
class LoanDetails(BaseModel):
    loan_exists: bool
    loan_amount: Optional[float] = None
    monthly_payment: Optional[float] = None
    loan_term: Optional[int] = None
    interest_rate: Optional[float] = None
    model_config = ConfigDict(from_attributes=True)
class Lifestyle(BaseModel):
    smoke: bool
    dine_out_frequency: str
    sports_hobbies: str
    model_config = ConfigDict(from_attributes=True)
class FinancialGoals(BaseModel):
    goal: str
    model_config = ConfigDict(from_attributes=True)
class UserFinancialData(BaseModel):
    user_id: int
    monthly_income: float
    expenses: 'Expenses'  # assuming you've defined Expenses elsewhere
    loan_details: 'LoanDetails'
    lifestyle: Lifestyle
    financial_goals: FinancialGoals
    model_config = ConfigDict(from_attributes=True)

@app.post("/financial_data")
def submit_financial_data(data: UserFinancialData, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == data.user_id).first() 
    if not user:
        return {"status": "0", "message": "User not found ! Please enter Vallid USER_ID","result":{}}
    user.monthly_income = data.monthly_income
    expenses = ExpensesDB(user_id=user.id, **data.expenses.dict()) 
    loan_details = LoanDetailsDB(user_id=user.id, **data.loan_details.dict()) 
    lifestyle = LifestyleDB(user_id=user.id, **data.lifestyle.dict()) 
    financial_goals = FinancialGoalsDB(user_id=user.id, **data.financial_goals.dict()) 
    db.add_all([expenses, loan_details, lifestyle, financial_goals])
    db.commit()
    return {"status": "1","message": "Financial data stored successfully","user_id": user.id,
        "financial_data": {
            "monthly_income": data.monthly_income,
            "expenses": data.expenses.dict(),
            "loan_details": data.loan_details.dict(),
            "lifestyle": data.lifestyle.dict(),
            "financial_goals": data.financial_goals.dict()} } 
#Get method
DB_PATH = "ryze_api_db"

# Column name mapping to match trained model
COLUMN_MAPPING = {
    "income": "Income",
    "rent": "Rent",
    "groceries": "Groceries",
    "transportation": "Transportation",
    "healthcare": "Healthcare",
    "dining_out": "Dining_Out",
    "shopping": "Shopping",
    "personal_care": "Personal_Care",
    "education": "Education",
    "electricity": "Electricity",
    "water": "Water",
    "insurance": "Insurance",}
DB_CONFIG = {
    "dbname": "ryze_api_db",
    "user": "postgres",
    "password": 'RGS@123', "host": "172.31.10.201",
    "port": "5432" }
# Fetch user data with correct column names
def fetch_user_data(user_id: int, db: Session):
    try:
        result = db.query(ExpensesDB).filter(ExpensesDB.user_id == user_id).first()
        if result:
            user_data = {COLUMN_MAPPING[key]: value for key, value in dict(result.__dict__).items() if key in COLUMN_MAPPING}
            return user_data
        else:
            raise HTTPException(status_code=404, detail=f"User data not found for user_id: {user_id}")
    except SQLAlchemyError as err:
        raise HTTPException(status_code=500, detail=f"Database error: {str(err)}")
def analyze_spending(user_data):
    income = user_data["Income"]
    user_df = pd.DataFrame([user_data])
    user_df["Total Spending"] = user_df.drop(columns=["Income"]).sum(axis=1)
    user_df["Spending Percentage"] = (user_df["Total Spending"] / income) * 100
    original_total_spending = user_df["Total Spending"][0]
    original_spending_percentage = user_df["Spending Percentage"][0]
    try:
        user_df[['Income', 'Total Spending', 'Spending Percentage']] = scaler.transform(
            user_df[['Income', 'Total Spending', 'Spending Percentage']])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during scaling: {str(e)}")
    try:
        anomaly_if = iso_forest.predict(user_df[['Income', 'Total Spending', 'Spending Percentage']])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during anomaly prediction: {str(e)}")
    spending_categories = {k: v for k, v in user_data.items() if k != "Income"}
    highest_spending_category = max(spending_categories, key=spending_categories.get) # type: ignore
    spending_percentage = user_df['Spending Percentage'][0]

    if spending_percentage <= 75:
        status = "Good"
        advice = "You're managing your spending well. Keep up the good financial habits!"
    elif spending_percentage > 75 and anomaly_if != -1:
        status = "Warning"
        advice = "You might be overspending. Consider cutting back on discretionary expenses."
    else:
        status = "Critical"
        advice = "Your spending pattern is unusual. Review your expenses to prevent financial risks."
    high_expenses = [
        cat for cat, value in spending_categories.items()
        if (cat in ['Rent', 'Education'] and value > (0.3 * income))
        or (cat not in ['Rent', 'Education'] and value > (0.2 * income))]
    overspending_alerts = [f"{cat} exceeds 20% of your income." for cat in high_expenses]
    return {
        "spending_analysis": {
            "status": status,
            "description": f"You are spending {original_spending_percentage:.2f}% of your income.",
            "highest_spending_category": highest_spending_category,
            "advice": advice,
            "overspending_alerts": overspending_alerts,
            "spending_details": {
                "Income": f"Your monthly income is ${income}. Consider allocating some to savings.",
                "Total Spending": f"You have spent ${original_total_spending:.2f} this month.",
                 "Spending Percentage": f"You are using {original_spending_percentage:.2f}% of your income.",
                "Breakdown": {
                    "Essentials": {
                        "Rent": f"You spend ${user_data['Rent']} on rent, a significant expense.",
                        "Groceries": f"You allocate ${user_data['Groceries']} to groceries.",
                        "Transportation": f"Your transportation costs are ${user_data['Transportation']}.",
                        "Healthcare": f"You spend ${user_data['Healthcare']} on healthcare.",
                        "Insurance": f"Your insurance costs are ${user_data['Insurance']}.",},
                    "Utilities": {
                        "Electricity": f"Electricity costs are ${user_data['Electricity']}.",
                        "Water": f"Your water bill is ${user_data['Water']}.",},
                    "Discretionary Spending": {
                        "Dining Out": f"Dining out expenses are ${user_data['Dining_Out']}.",
                        "Shopping": f"Shopping costs are ${user_data['Shopping']}.",
                        "Personal Care": f"Personal care expenses are ${user_data['Personal_Care']}.",
                        "Education": f"Education costs amount to ${user_data['Education']}.", }}}}}
        
class Spending_Request(BaseModel):
    user_id: int
@app.get("/predict_spending_behavior")
def predict_spending_behavior(request: Spending_Request, db: Session = Depends(get_db)):
    try:
        user_data = fetch_user_data(request.user_id, db)
        result = analyze_spending(user_data)
        result["user_id"] = request.user_id
        return result
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/View_Profile")
def view_profile(request: UserProfileRequest = Body(...), db: Session = Depends(get_db)):
    user = db.query(User).options(joinedload(User.expenses),joinedload(User.loan_details),
        joinedload(User.lifestyle),joinedload(User.financial_goals)).filter(User.id == request.user_id).first()
    if not user:
        return {"status": "0", "message": "User not found", "result": {}}
    return { "status": "1","message": "User profile fetched successfully",
        "result": { "user_name": user.user_name, "email": user.email,"mobile": user.mobile, "profile_image": user.profile_image,
            "created_at": str(user.created_at),
            "updated_at": str(user.updated_at),
            "expenses": [filter_out_id(e.__dict__) for e in user.expenses],
            "loan_details": [filter_out_id(l.__dict__) for l in user.loan_details],
            "lifestyle": filter_out_id(user.lifestyle.__dict__) if user.lifestyle else {},
            "financial_goals": filter_out_id(user.financial_goals.__dict__) if user.financial_goals else {}}}
def filter_out_id(data: dict) -> dict:
    return {k: v for k, v in data.items() if k != "id" and not k.startswith("_sa_")}

@app.post("/Edit_profile")
def edit_profile(
    user_id: int = Form(...),
    user_name: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    mobile: Optional[str] = Form(None),
    profile_image: Optional[UploadFile] = File(None),
    expenses: Optional[str] = Form(None),  
    loan_details: Optional[str] = Form(None),  
    lifestyle: Optional[str] = Form(None),  
    financial_goals: Optional[str] = Form(None), 
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return {"status":"0","message":"User not found","results":{}}
    # Update basic fields
    if user_name:
        user.user_name = user_name
    if email:
        user.email = email
    if mobile:
        user.mobile = mobile
    # Handle profile image upload
    if profile_image:
        image_path = os.path.join(MEDIA_DIR, profile_image.filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(profile_image.file, buffer)
        user.profile_image = profile_image.filename
    # Update expenses
    if expenses and expenses.strip() != "":
        try:
            exp_data = json.loads(expenses)
            if isinstance(exp_data, list):
                exp_list = [Expenses(**e) for e in exp_data]
                db.query(ExpensesDB).filter(ExpensesDB.user_id == user_id).delete()
                for exp in exp_list:
                    db.add(ExpensesDB(user_id=user_id, **exp.dict()))
            else:
                return {"status":"0","message":"Expenses should be a list","results":{e}}
        except json.JSONDecodeError as e:
            return {"status":"0","message":"Invalid JSON in Expenses","results":{}}
    # Update loan details
    if loan_details:
        try:
            loan_list = [LoanDetails(**l) for l in json.loads(loan_details)]
            db.query(LoanDetailsDB).filter(LoanDetailsDB.user_id == user_id).delete()
            for loan in loan_list:
                db.add(LoanDetailsDB(user_id=user_id, **loan.dict()))
        except Exception as e:
            return {"status":"0","message":"Invalid loan details","results":{}}
    # Update lifestyle
    if lifestyle:
        try:
            life_data = Lifestyle(**json.loads(lifestyle))
            if user.lifestyle:
                for key, value in life_data.dict().items():
                    setattr(user.lifestyle, key, value)
            else:
                db.add(LifestyleDB(user_id=user_id, **life_data.dict()))
        except Exception as e:
            return {"status":"0","message":"Invalid lifestyle data","results":{}}
    # Update financial goals
    if financial_goals:
        try:
            goal_data = FinancialGoals(**json.loads(financial_goals))
            if user.financial_goals:
                for key, value in goal_data.dict().items():
                    setattr(user.financial_goals, key, value)
            else:
                db.add(FinancialGoalsDB(user_id=user_id, **goal_data.dict()))
        except Exception as e:
            return {"status":"0","message":"Invallid financial goals data","results":{}} 
    db.commit()
    return {"status": "1", "message": "User profile updated successfully"}
class DashboardRequest(BaseModel):
    user_id: int
class CategoryStats(BaseModel):
    category: str
    total: float
    model_config = ConfigDict(from_attributes=True)
class CategoryStat(BaseModel):
    category: str
    total: float  
    model_config = ConfigDict(from_attributes=True)
class MonthlyTrend(BaseModel):
    month: str
    total: float
    model_config = ConfigDict(from_attributes=True)
class MonthlyExpenseTrend(BaseModel):
    month: str
    total: float  
    model_config = ConfigDict(from_attributes=True)
class DashboardResponse(BaseModel):
    total_spending: float
    income : float
    spending_percentage: float
    expenses_breakdown: Dict[str, float]
    category_stats: List[CategoryStats]
    monthly_trend: List[MonthlyTrend]
    loan_details: Optional[LoanDetails]
    lifestyle: Optional[Lifestyle]
    financial_goals: Optional[FinancialGoals]
    model_config = ConfigDict(from_attributes=True)
def get_current_user(user_id: int, db: Session = Depends(get_db)) -> User:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return {"status":"0","message":"User not found","results":{}}
    return user    
@app.post("/dashboard", response_model=DashboardResponse)
def get_dashboard(data: DashboardRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == data.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    expense = db.query(ExpensesDB).filter_by(user_id=user.id).first()
    if not expense:
        raise HTTPException(status_code=404, detail="Expenses not found")
    fields = [
        "rent", "groceries", "transportation", "healthcare", "dining_out",
        "shopping", "personal_care", "education", "electricity", "water", "insurance"]
    
    breakdown = {}
    total = 0
    for field in fields:
        val = getattr(expense, field, 0) or 0
        breakdown[field] = val
        total += val

    income = expense.income or 0
    spending_percent = round((total / income) * 100, 2) if income else 0
    category_stats = [
        CategoryStat(category=key, total=value)
        for key, value in breakdown.items()]
    monthly_trend = [
        MonthlyExpenseTrend(month=month, total=0.0)
        for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] ]

    loan = db.query(LoanDetailsDB).filter_by(user_id=user.id).first()
    lifestyle = db.query(LifestyleDB).filter_by(user_id=user.id).first()
    goals = db.query(FinancialGoalsDB).filter_by(user_id=user.id).first()

    return DashboardResponse(
        total_spending=total,
        income=income,
        spending_percentage=spending_percent,
        expenses_breakdown=breakdown,
        category_stats=category_stats,
        monthly_trend=monthly_trend,
        loan_details=LoanDetails.from_orm(loan) if loan else None,
        lifestyle=Lifestyle.from_orm(lifestyle) if lifestyle else None,
        financial_goals=FinancialGoals.from_orm(goals) if goals else None )  
class GoalCreate(BaseModel):
    user_id: int
    goal_title: str
    amount: str
    current_amount: str
    deadline: str  
@app.post("/add_new_goal")
def add_goal(request: GoalCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == request.user_id).first()
    if not user:
        return {"status":"0","message":"User not found","results":{}}
    new_goal = FinancialGoalsDB(
        user_id=request.user_id,
        goal_title=request.goal_title,
        amount=request.amount,
        current_amount=request.current_amount,
        deadline=request.deadline)
    db.add(new_goal)
    db.commit()
    db.refresh(new_goal)
    return {"status":"1","message": "Goal added successfully", "results": {
        "goal_title": new_goal.goal_title,
        "amount": new_goal.amount,
        "current_amount": new_goal.current_amount,
        "deadline": new_goal.deadline }}
class UserIDRequest(BaseModel):
    user_id: int    
@app.post("/all_goals")
def all_goals(request: UserIDRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == request.user_id).first()
    if not user:
        return {"status": "0", "message": "User not found", "results": []}
    goals = db.query(FinancialGoalsDB).filter(FinancialGoalsDB.user_id == request.user_id).all()
    results = [ {"goal_title": g.goal_title,"amount": g.amount,"current_amount": g.current_amount,"deadline": g.deadline}
        for g in goals]
    return {
        "status": "1",
        "message": "Goals fetched successfully",
        "results": results}
###################################################################
import joblib
import numpy as np
import ollama
# Load the model
try:
    model = joblib.load("model/savings_predictor_model.pkl")
except Exception as e:
    print("Failed to load model:", e)
    raise e
# Request schema
class FinancialData(BaseModel):
    user_id : int
    income: float
    rent: float
    groceries: float
    transportation: float
    healthcare: float
    food: float
    shopping: float
    personal_care: float
    education: float
    electricity: float
    water: float
    insurance: float
    loan_exists: bool
    loan_amount: float
    monthly_emi: float
    loan_term_years: float
    interest_rate: float
    smoke: bool
    dine_out_freq: str
    hobbies: str
    goal: str
    goal_amount: float
    goal_timeframe: int
    
def build_prompt(data: FinancialData, savings: float) -> str:
    return f"""
You are a smart financial assistant.

The user has the following financial goal:
- Goal type: {data.goal}
- Target amount: ₹{data.goal_amount}
- Timeframe: {data.goal_timeframe} years

Their predicted monthly savings is ₹{savings:.0f}.

Other details:
- Monthly income: ₹{data.income}
- Expenses: rent ₹{data.rent}, groceries ₹{data.groceries}, food ₹{data.food}, transportation ₹{data.transportation}, healthcare ₹{data.healthcare}
- Loan: Exists: {"Yes" if data.loan_exists else "No"}, Amount: ₹{data.loan_amount}, EMI: ₹{data.monthly_emi}, Interest rate: {data.interest_rate}%
- Lifestyle: Smokes: {"Yes" if data.smoke else "No"}, Dines out: {data.dine_out_freq}, Hobbies: {data.hobbies}

Please provide:
- Whether the goal is achievable or not
- A savings plan (monthly)
"""
def get_mistral_response(prompt: str) -> str:
    try:
        response = ollama.chat(model='mistral', messages=[
            {"role": "system", "content": "You are a helpful financial advisor."},
            {"role": "user", "content": prompt}])
        return response['message']['content']
    except Exception as e:
        return f"❌ Mistral error: {e}"

@app.post("/predict_savings/")
def predict_savings(request: FinancialData, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == request.user_id).first()
    if not user:
        return {"status": "0", "message": "User not found", "results": []}
    try:
        loan_exists = 1 if request.loan_exists else 0
        smoke = 1 if request.smoke else 0
        X = np.array([[  # Must match model training order
            request.income, request.rent, request.groceries, request.transportation,
            request.healthcare, request.food, request.shopping, request.personal_care,
            request.education, request.electricity, request.water, request.insurance,
            request.loan_amount, request.monthly_emi, request.loan_term_years,
            request.interest_rate, smoke]])
        predicted_savings = model.predict(X)[0]
        prompt = build_prompt(request, predicted_savings)
        insight = get_mistral_response(prompt)
        return {
            "predicted_monthly_savings": round(predicted_savings, 2),"ai_insight": insight  }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    #######
class LoanResponse(BaseModel):
    user_id: int
    loan_number: str
    loan_type: str
    loan_amount: float
    monthly_payment: float
    loan_term: int
    interest_rate: float
    start_date: datetime
    status: str
    created_at: datetime
    updated_at: datetime    
@app.post("/add_new_loan")
def add_loan(request: LoanResponse, db: Session = Depends(get_db)):
    # Check if the user exists
    user = db.query(User).filter(User.id == request.user_id).first()
    if not user:
        return {"status": "0", "message": "User not found", "results": {}}
    # Generate loan number (e.g., LN0001)
    last_loan = db.query(LoanDetailsDB).order_by(LoanDetailsDB.id.desc()).first()
    next_id = (last_loan.id + 1) if last_loan else 1
    
    # Create and store new loan
    new_loan = LoanDetailsDB(
        user_id=request.user_id, 
        loan_number=request.loan_number,
        loan_type=request.loan_type,
        loan_amount=request.loan_amount,
        monthly_payment=request.monthly_payment,
        loan_term=request.loan_term,
        interest_rate=8.5,  # default interest
        start_date=datetime.utcnow(),
        status="active",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow())
    db.add(new_loan)
    db.commit()
    db.refresh(new_loan)
    return {
        "status": "1",
        "message": "Loan added successfully",
        "results": {
            "loan_number": new_loan.loan_number,
            "loan_type": new_loan.loan_type,
            "loan_amount": new_loan.loan_amount,
            "monthly_payment": new_loan.monthly_payment,
            "loan_term": new_loan.loan_term,
            "interest_rate": new_loan.interest_rate,
            "start_date": new_loan.start_date,
            "status": new_loan.status }}
class LoanIDRequest(BaseModel):
    user_id: int
@app.post("/all_loans")
def all_loans(request: LoanIDRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == request.user_id).first()
    if not user:
        return {"status": "0", "message": "User not found", "results": {}}
    loans = db.query(LoanDetailsDB).filter(LoanDetailsDB.user_id == request.user_id).all()
    results = [
        {
            "loan_number": loan.loan_number,
            "loan_type": loan.loan_type,
            "loan_amount": loan.loan_amount,
            "monthly_payment": loan.monthly_payment,
            "loan_term": loan.loan_term,
            "interest_rate": loan.interest_rate,
            "start_date": loan.start_date,
            "status": loan.status } for loan in loans]
    return {
        "status": "1",
        "message": "Loans fetched successfully",
        "results": results }
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
