package com.ollama.ollama.controller;

import jakarta.servlet.RequestDispatcher;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.ui.Model;
import org.springframework.boot.webmvc.error.ErrorController;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

@Controller
public class CustomErrorController implements ErrorController {

    @RequestMapping(value="/400", method=RequestMethod.POST)
    public String handleError400POST(HttpServletRequest request, Model model) {
        return "/static/400";
    }

    @RequestMapping(value="/400", method=RequestMethod.GET)
    public String handleError400GET(HttpServletRequest request, Model model) {
        return "/static/400";
    }

    @RequestMapping(value="/404", method=RequestMethod.POST)
    public String handleError404POST(HttpServletRequest request, Model model) {
        return "/static/404";
    }

    @RequestMapping(value="/404", method=RequestMethod.GET)
    public String handleError404GET(HttpServletRequest request, Model model) {
        return "/static/404";
    }

    @RequestMapping(value="/500", method=RequestMethod.POST)
    public String handleError500POST(HttpServletRequest request, Model model) {
        return "/static/500";
    }

    @RequestMapping(value="/500", method=RequestMethod.GET)
    public String handleError500GET(HttpServletRequest request, Model model) {
        return "/static/500";
    }

    @RequestMapping(value="/error", method=RequestMethod.POST)
    public String handleErrorPOST(HttpServletRequest request, RedirectAttributes redirectAttributes, Model model) {
        return this.handleError(request, redirectAttributes, model);
    }

    @RequestMapping(value="/error", method=RequestMethod.GET)
    public String handleErrorGET(HttpServletRequest request, RedirectAttributes redirectAttributes, Model model) {
        return this.handleError(request, redirectAttributes, model);
    }

    public String getErrorPath() {
        return "/error/common";
    }

    private String handleError(HttpServletRequest request, RedirectAttributes redirectAttributes, Model model) {
        // Retrieve the error message
        Object errorMessage = request.getAttribute(RequestDispatcher.ERROR_MESSAGE);

        // Retrieve the HTTP status code
        Object status = request.getAttribute(RequestDispatcher.ERROR_STATUS_CODE);

        int statusCode = (status != null) ? Integer.parseInt(status.toString()) :
                HttpStatus.INTERNAL_SERVER_ERROR.value();
        String message = (errorMessage != null) ? errorMessage.toString() : "No message available";

        model.addAttribute("status", statusCode);
        model.addAttribute("message", message);

        String url;

        if (statusCode == 400) {
            url = "/static/400";
        } else if (statusCode == 404) {
            url = "/static/404";
        } else if (statusCode == 500) {
            url = "/static/500";
        } else {
            url = "/error/common";
        }

        return url;
    }
}
