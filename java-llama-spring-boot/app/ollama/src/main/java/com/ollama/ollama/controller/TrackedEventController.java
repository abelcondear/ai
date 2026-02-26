package com.ollama.ollama.controller;

import com.ollama.ollama.component.TaskPromptStore;
import com.ollama.ollama.component.TaskStatusStore;
import com.ollama.ollama.configuration.TaskStatus;
import com.ollama.ollama.service.TrackedEventService;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.ui.Model;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

@Controller
public class TrackedEventController {
    private final TrackedEventService service;
    private final TaskStatusStore statusStore;
    private final TaskPromptStore promptStore;

    private List<String> prompts = new ArrayList<>(List.of());
    private List<String> tasksId = new ArrayList<>(List.of());
    private List<Map<String, String>> data = new ArrayList<>(List.of());
    
    public TrackedEventController(TrackedEventService service, TaskStatusStore statusStore,
                                  TaskPromptStore promptStore) {
        this.service = service;
        this.statusStore = statusStore;
        this.promptStore = promptStore;
    }

    @GetMapping("/home")
    public String home(Model model) {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss");
        model.addAttribute("currentDate", LocalDateTime.now().format(formatter));

        String url = "/public/home";
        String prompt;

        prompt = "Hello";
        if (!this.checkExist(prompt)) {
            this.addPrompt(model, prompt);
        }

        prompt = "Hi";
        if (!this.checkExist(prompt)) {
            this.addPrompt(model, prompt);
        }

        prompt = "Buenos días";
        if (!this.checkExist(prompt)) {
            this.addPrompt(model, prompt);
        }

        prompt = "Guten Tag, mein Freund";
        if (!this.checkExist(prompt)) {
            this.addPrompt(model, prompt);
        }

        prompt = "Tell me who is it?";
        if (!this.checkExist(prompt)) {
            this.addPrompt(model, prompt);
        }

        this.updateData();
        this.loadPrompts(model);

        return url;
    }

    private void addPrompt(Model model, String prompt) {
        prompts.add(prompt);

        String taskId = service.triggerTrackedEvent(prompt);

        tasksId.add(taskId);

        Map<String, String> d = new HashMap<>();

        d.put("taskId", taskId);
        d.put("status", this.translate(statusStore.readStatus(taskId)));

        data.add(d);

        model.addAttribute("data", data);
    }

    private String translate(TaskStatus status) {
        String result = "";

        if (status == TaskStatus.PENDING) { result = "Pending"; }
        if (status == TaskStatus.RUNNING) { result = "Running"; }
        if (status == TaskStatus.COMPLETED) { result = "Completed"; }
        if (status == TaskStatus.FAILED) { result = "Failed"; }

        return result;
    }

    private Boolean checkExist(String prompt) {
        Boolean result = false;

        for (String taskId: tasksId) {
            if (Objects.equals(prompt, promptStore.readPrompt(taskId).getPrompt())) {
                result = true; // move to next taskId. because, taskId was already created
                break;
            }
        }

        return result;
    }

    private void updateData() {
        for (String taskId: tasksId) {
            Integer index = 0;

            for (Map<String, String> item : data) {
                if (Objects.equals(item.get("taskId"), taskId)) {
                    data.get(index).put("prompt", promptStore.readPrompt(taskId).getPrompt());
                    data.get(index).put("response",
                            promptStore.readPrompt(taskId).getResponse().isEmpty() ?
                            "":
                            String.join(
                                " ",
                                promptStore.readPrompt(taskId).getResponse()
                            ).replace(
                                    "\n", "<br/>"
                            ).replace(
                                    "\n\n", "<br/><br/>"
                            ).replace(
                                    "\"", "&quot;"
                            )
                    );
                    data.get(index).put("status", this.translate(statusStore.readStatus(taskId)));
                }

                index++;
            }
        }
    }

    private void loadPrompts(Model model) {
        model.addAttribute("data", data);
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
        String taskId = "";

        try {
            taskId = service.triggerTrackedEvent(prompt);
            model.addAttribute("taskId", taskId);
            model.addAttribute("status", statusStore.readStatus(taskId));
            url = "/public/prompt";
        } catch (RuntimeException e) {
            statusStore.setStatus(taskId, TaskStatus.FAILED);

            model.addAttribute("message", e.getMessage());
            url = "/error/common";
        }

        return url;
    }

    @GetMapping("/trigger-response/{taskId}")
    public String triggerResponse(@PathVariable("taskId") String taskId, Model model) {
        String url = "";

        if (statusStore.readStatus(taskId) == TaskStatus.COMPLETED) {
            try {
                model.addAttribute("taskId", taskId);
                model.addAttribute("status", statusStore.readStatus(taskId));

                model.addAttribute("prompt", promptStore.readPrompt(taskId).getPrompt());
                model.addAttribute
                        (
                                "response",
                                String.join(" ", promptStore.readPrompt(taskId).getResponse()).replace("\n", "<br/>")
                        );

                url = "/public/response";
            } catch (RuntimeException e) {
                url = "/error";
            }
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
            url = "/error";
        }

        return url;
    }
}