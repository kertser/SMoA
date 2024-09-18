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
    logging.error(f"OpenAI API error: {str(e)}", exc_info=True)

    error_mapping = {
        AuthenticationError: ("Authentication error with OpenAI API. Please check your API key.", 401),
        PermissionDeniedError: ("You do not have permission to perform this action. Please check your OpenAI account permissions.", 403),
        RateLimitError: ("Rate limit exceeded. Please wait before making more requests.", 429),
        BadRequestError: (f"Invalid request to OpenAI API: {str(e)}", 400),
        APIConnectionError: ("Failed to connect to OpenAI API. Please check your network connection.", 502),
        Timeout: ("Request to OpenAI API timed out. Please try again later.", 504),
    }

    error_message, status_code = error_mapping.get(type(e), (f"An error occurred with the OpenAI API: {str(e)}", 500))
    return create_error_response(error_message, status_code)

def create_error_response(message, status_code):
    """Create a Flask JSON response with an error message and status code."""
    response = jsonify({"error": {"message": message}})
    response.status_code = status_code
    return response

def handle_request_error(e):
    """Handle errors related to invalid requests."""
    logging.error(f"Request error: {str(e)}", exc_info=True)
    return create_error_response(str(e), 400)

def handle_internal_error(e):
    """Handle unexpected internal server errors."""
    logging.error(f"Internal server error: {str(e)}", exc_info=True)
    return create_error_response("An internal server error occurred.", 500)