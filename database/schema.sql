-- 1. BED MASTER TABLE
CREATE TABLE beds (
    bed_id VARCHAR(10) PRIMARY KEY, -- e.g., 'ICU-A-01'
    unit_name VARCHAR(50),
    bed_type VARCHAR(20) CHECK (bed_type IN ('ICU', 'General', 'Isolation')),
    is_operational BOOLEAN DEFAULT TRUE,
    current_status VARCHAR(20) DEFAULT 'Available' -- Available, Occupied, Cleaning
);

-- 2. ACTIVE PATIENTS & ADMISSIONS
CREATE TABLE patients (
    patient_id VARCHAR(20) PRIMARY KEY,
    mrn VARCHAR(20) UNIQUE,
    age INT,
    sex VARCHAR(10),
    admission_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    clinical_data JSONB, -- Stores {sofa: 4, cci: 2, ...}
    status VARCHAR(20) -- Admitted, Discharged, Transferred
);

-- 3. AI PREDICTIONS LOG
CREATE TABLE predictions (
    prediction_id SERIAL PRIMARY KEY,
    patient_id VARCHAR(20) REFERENCES patients(patient_id),
    model_version VARCHAR(20),
    predicted_los FLOAT,
    confidence_interval_lower FLOAT,
    confidence_interval_upper FLOAT,
    predicted_discharge_date TIMESTAMP,
    shap_factors JSONB, -- Stores top contributing features
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. BED OCCUPANCY (Actual & Forecast)
CREATE TABLE occupancy (
    log_id SERIAL PRIMARY KEY,
    bed_id VARCHAR(10) REFERENCES beds(bed_id),
    patient_id VARCHAR(20) REFERENCES patients(patient_id),
    start_time TIMESTAMP,
    end_time_estimated TIMESTAMP,
    end_time_actual TIMESTAMP
);