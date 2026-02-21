package com.ollama.ollama.controller;

import jakarta.servlet.RequestDispatcher;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.boot.webmvc.error.ErrorController;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import java.util.HashMap;
import java.util.Map;

@Controller
@RequestMapping("/error")
public class CustomErrorController implements ErrorController {

    @RequestMapping
    @ResponseBody
    public ResponseEntity<Map<String, Object>> handleError(HttpServletRequest request) {
        // Retrieve the error message
        Object errorMessage = request.getAttribute(RequestDispatcher.ERROR_MESSAGE);

        // Retrieve the HTTP status code
        Object status = request.getAttribute(RequestDispatcher.ERROR_STATUS_CODE);
        Integer statusCode = status != null ? Integer.valueOf(status.toString()) : HttpStatus.INTERNAL_SERVER_ERROR.value();

        Map<String, Object> errorDetails = new HashMap<>();
        errorDetails.put("status", statusCode);
        errorDetails.put("message", errorMessage != null ? errorMessage.toString() : "No message available");
        // Add other details if needed, e.g., timestamp, path

        return new ResponseEntity<>(errorDetails, HttpStatus.valueOf(statusCode));
    }

    public String getErrorPath() {
        return "/error";
    }
}
