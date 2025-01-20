"""

import sys


from custom_logger import logger

def error_message_details(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        #super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail)
        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message 

"""


import sys
from custom_logger import logger

def error_message_details(error, error_detail: sys):
    exc_info = error_detail.exc_info()
    error_message = str(error)  # Default to just the error message if no traceback

    if exc_info and exc_info[2]:  # Check if traceback is available
        _, _, exc_tb = exc_info
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
            file_name, exc_tb.tb_lineno, str(error)
        )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        self.error_message = error_message_details(error_message, error_detail)
        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message
