from flask import jsonify
from openai import OpenAIError, APIError, APITimeoutError, RateLimitError, APIConnectionError, BadRequestError, AuthenticationError, APIStatusError

def handle_openai_error(e):
    if isinstance(e, AuthenticationError):
        if "Invalid Authentication" in str(e):
            return create_error_response("Invalid Authentication. Ensure the correct API key and requesting organization are being used.", 401)
        elif "Incorrect API key provided" in str(e):
            return create_error_response("Incorrect API key provided. Ensure the API key used is correct, clear your browser cache, or generate a new one.", 401)
        elif "You must be a member of an organization to use the API" in str(e):
            return create_error_response("You must be a member of an organization to use the API. Contact us to get added to a new organization or ask your organization manager to invite you to an organization.", 401)
        else:
            return create_error_response("Authentication error with OpenAI API.", 401)
    elif isinstance(e, BadRequestError):
        if "Country, region, or territory not supported" in str(e):
            return create_error_response("Country, region, or territory not supported. Please see our documentation for more information.", 403)
        else:
            return create_error_response("Invalid request to OpenAI API.", 400)
    elif isinstance(e, RateLimitError):
        if "You exceeded your current quota" in str(e):
            return create_error_response("You exceeded your current quota. Please check your plan and billing details.", 429)
        else:
            return create_error_response("Rate limit reached for requests. Please pace your requests.", 429)
    elif isinstance(e, APIConnectionError):
        return create_error_response("Failed to connect to OpenAI API.", 503)
    elif isinstance(e, APITimeoutError):
        return create_error_response("Request to OpenAI API timed out.", 504)
    elif isinstance(e, APIStatusError):
        if e.status_code == 500:
            return create_error_response("The server had an error while processing your request. Please retry your request after a brief wait and contact OpenAI if the issue persists.", 500)
        elif e.status_code == 503:
            return create_error_response("The engine is currently overloaded. Please try again later.", 503)
        else:
            return create_error_response(f"OpenAI API returned an unexpected status: {e.status_code}", e.status_code)
    elif isinstance(e, APIError):
        return create_error_response("Error in OpenAI API.", 500)
    else:
        return create_error_response("An unexpected error occurred.", 500)

def create_error_response(message, status_code):
    response = jsonify({"error": {"message": message}})
    response.status_code = status_code
    return response

def handle_request_error(e):
    return create_error_response(str(e), 400)

def handle_internal_error(e):
    return create_error_response("An internal server error occurred.", 500)
