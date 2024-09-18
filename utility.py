from flask import jsonify
import openai
from openai import (
    OpenAIError,
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
    APIConnectionError,
    Timeout,
)

import logging

def handle_openai_error(e):
    """Handle exceptions from the OpenAI API and return appropriate Flask responses."""
    # Log the exception details
    logging.error(f"OpenAI API error: {str(e)}")

    if isinstance(e, AuthenticationError):
        return create_error_response("Authentication error with OpenAI API. Please check your API key.", 401)
    elif isinstance(e, PermissionError):
        return create_error_response("You do not have permission to perform this action. Please check your OpenAI account permissions.", 403)
    elif isinstance(e, RateLimitError):
        return create_error_response("Rate limit exceeded. Please wait before making more requests.", 429)
    elif isinstance(e, InvalidRequestError):
        return create_error_response(f"Invalid request to OpenAI API: {str(e)}", 400)
    elif isinstance(e, ServiceUnavailableError):
        return create_error_response("OpenAI service is currently unavailable. Please try again later.", 503)
    elif isinstance(e, APIConnectionError):
        return create_error_response("Failed to connect to OpenAI API. Please check your network connection.", 502)
    elif isinstance(e, Timeout):
        return create_error_response("Request to OpenAI API timed out. Please try again later.", 504)
    elif isinstance(e, TryAgain):
        return create_error_response("OpenAI API is currently overloaded. Please try again later.", 503)
    elif isinstance(e, OpenAIError):
        # General OpenAI error
        return create_error_response(f"An error occurred with the OpenAI API: {str(e)}", 500)
    else:
        # Unexpected exception
        logging.error("An unexpected error occurred.", exc_info=True)
        return create_error_response("An unexpected error occurred.", 500)

def create_error_response(message, status_code):
    """Create a Flask JSON response with an error message and status code."""
    response = jsonify({"error": {"message": message}})
    response.status_code = status_code
    return response

def handle_request_error(e):
    """Handle errors related to invalid requests."""
    logging.error(f"Request error: {str(e)}")
    return create_error_response(str(e), 400)

def handle_internal_error(e):
    """Handle unexpected internal server errors."""
    logging.error(f"Internal server error: {str(e)}", exc_info=True)
    return create_error_response("An internal server error occurred.", 500)
