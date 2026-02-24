package com.ollama.ollama.controller;

import com.ollama.ollama.component.TaskPromptStore;
import com.ollama.ollama.component.TaskStatusStore;
import com.ollama.ollama.configuration.TaskStatus;
import com.ollama.ollama.service.TrackedEventService;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.ui.Model;

import java.util.Objects;

@Controller
public class TrackedEventController {
    private final TrackedEventService service;
    private final TaskStatusStore statusStore;
    private final TaskPromptStore promptStore;

    public TrackedEventController(TrackedEventService service, TaskStatusStore statusStore,
                                  TaskPromptStore promptStore) {
        this.service = service;
        this.statusStore = statusStore;
        this.promptStore = promptStore;
    }

    @GetMapping("/home")
    public String home(Model model) {
        return "/public/home";
    }

    @GetMapping("/prompt")
    public String prompt(Model model) {
        return "/public/prompt";
    }

    @GetMapping("/response")
    public String response(Model model) {
        return "/public/response";
    }

    @GetMapping("/status")
    public String status(Model model) {
        return "/public/status";
    }


    @GetMapping("/trigger-tracked/{prompt}")
    public String triggerTracked(@PathVariable("prompt") String prompt, Model model) {
        String url;

        try {
            String taskId = service.triggerTrackedEvent(prompt);
            model.addAttribute("taskId", taskId);
            model.addAttribute("status", statusStore.readStatus(taskId));
            url = "/public/prompt";
        } catch (RuntimeException e) {
            model.addAttribute("message", e.getMessage());
            url = "/error/common";
        }

        return url;
    }

    @GetMapping("/trigger-response/{taskId}")
    public String triggerResponse(@PathVariable("taskId") String taskId, Model model) {
        String result;
        String url;

        if (statusStore.readStatus(taskId) == TaskStatus.COMPLETED) {
            try {
                result = "Task " + taskId + " - prompt: " + promptStore.readPrompt(taskId).getPrompt()  + " - " +
                        "response: " + String.join(" ", promptStore.readPrompt(taskId).getResponse());

                model.addAttribute("taskId", taskId);
                model.addAttribute("status", statusStore.readStatus(taskId));

                model.addAttribute("prompt", promptStore.readPrompt(taskId).getPrompt());
                model.addAttribute
                (
                        "response",
                        String.join(" ", promptStore.readPrompt(taskId).getResponse()).replace("\n","<br/>")
                );

                url = "/public/response";
            } catch (RuntimeException e) {
                model.addAttribute("message", e.getMessage());
                url = "/error/common";
            }
        }
        else {
            model.addAttribute("taskId", taskId);
            model.addAttribute("status", statusStore.readStatus(taskId));
            url = "/public/response";
        }

        return url;
    }

    @GetMapping("/status/{taskId}")
    public String getStatus(@PathVariable("taskId") String taskId, Model model) {
        String url;

        try {
            model.addAttribute("taskId", taskId);
            model.addAttribute("status", statusStore.readStatus(taskId));

            url = "/public/status";
        } catch (RuntimeException e) {
            model.addAttribute("message", e.getMessage());
            url = "/error/common";
        }

        return url;
    }
}