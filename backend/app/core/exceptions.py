from fastapi import HTTPException, status


class SMSShieldException(HTTPException):
    """Base exception for SMS Spam Shield"""
    
    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: str = "An error occurred",
    ):
        super().__init__(status_code=status_code, detail=detail)


class ModelNotLoadedException(SMSShieldException):
    def __init__(self, model_name: str):
        detail = f"Model '{model_name}' is not loaded. Please train the model first."
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail
        )


class PredictionException(SMSShieldException):
    def __init__(self, model_name: str, error: str):
        detail = f"Prediction failed for model '{model_name}': {error}"
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )