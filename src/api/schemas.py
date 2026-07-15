from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    Temperature: float = Field(..., description="Temperature in Celsius", example=25.5)
    Humidity: float = Field(..., description="Humidity percentage", ge=0, le=100, example=60)
    Wind_Speed: float = Field(..., description="Wind speed in m/s", ge=0, example=1.2)
    Visibility: int = Field(..., description="Visibility in 10m", ge=0, example=2000)
    Solar_Radiation: float = Field(..., description="Solar Radiation in MJ/m2", ge=0, example=1.5)
    Rainfall: float = Field(..., description="Rainfall in mm", ge=0, example=0.0)
    Snowfall: float = Field(..., description="Snowfall in cm", ge=0, example=0.0)
    
    Hour: int = Field(..., description="Hour of the day (0-23)", ge=0, le=23, example=8)
    Month: int = Field(..., description="Month of the year (1-12)", ge=1, le=12, example=7)
    
    Season: str = Field(..., description="Spring, Summer, Autumn, Winter", example="Summer")
    Day_Type: str = Field(..., description="Work or Leisure", example="Work")
    Holiday: str = Field(..., description="Holiday or No Holiday", example="No Holiday")
    Functioning_Day: str = Field(..., description="Yes or No", example="Yes")

class PredictionResponse(BaseModel):
    expected_bike_demand: int
    message: str