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

import java.nio.charset.StandardCharsets;
import java.nio.ByteBuffer;

@Controller
public class TrackedEventController {
    private final TrackedEventService service;
    private final TaskStatusStore statusStore;
    private final TaskPromptStore promptStore;

    private Model model;

    private List<String> prompts = new ArrayList<>(List.of());
    private List<String> tasksId = new ArrayList<>(List.of());
    private List<Map<String, String>> data = new ArrayList<>(List.of());
    
    public TrackedEventController(TrackedEventService service, TaskStatusStore statusStore,
                                  TaskPromptStore promptStore) {
        this.service = service;
        this.statusStore = statusStore;
        this.promptStore = promptStore;
        //this.model = model;
    }

    @GetMapping("/home")
    public String home(Model model) {
        this.model = model;

        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss");
        this.model.addAttribute("currentDate", LocalDateTime.now().format(formatter));

        String url = "/public/home";
//        String prompt;
//
//        prompt = "Hello";
//        if (!this.checkExist(prompt)) {
//            this.addPrompt(this.model, prompt);
//        }
//
//        prompt = "Buenos días, ¿Cómo estás?.";
//        if (!this.checkExist(prompt)) {
//            this.addPrompt(this.model, prompt);
//        }
//
//        prompt = "Guten Tag, mein Freund";
//        if (!this.checkExist(prompt)) {
//            this.addPrompt(this.model, prompt);
//        }
//
//        prompt = "Tell me who is it?";
//        if (!this.checkExist(prompt)) {
//            this.addPrompt(this.model, prompt);
//        }
//
//        prompt = "Como vai?";
//        if (!this.checkExist(prompt)) {
//            this.addPrompt(this.model, prompt);
//        }
//
        this.updateData();
        this.loadPrompts(this.model);

        return url;
    }

    @PostMapping("/home")
    public String handleForm(
            @PathVariable("editPromptEN") @RequestParam("editPromptEN") String editPromptEN,
            @PathVariable("editPromptES") @RequestParam("editPromptES") String editPromptES,
            @PathVariable("editPromptFR") @RequestParam("editPromptFR") String editPromptFR,
            @PathVariable("editPromptDT") @RequestParam("editPromptDT") String editPromptDT,
            @PathVariable("editPromptPT") @RequestParam("editPromptPT") String editPromptPT,
            Model model
    ) {
        // service was temporally removed
//        System.out.println();
//        System.out.printf("editPromptES: %s", editPromptES);
//        System.out.println();
//
//        this.model = model;
//
//        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss");
//        this.model.addAttribute("currentDate", LocalDateTime.now().format(formatter));
//
//        String url = "/public/home";
//        String prompt;
//
//        // prompt received successfully by POST method
//        prompt = editPromptES;
//        if (!this.checkExist(prompt)) {
//            this.addPrompt(this.model, prompt);
//        }

//      ----------------------------------
//      testing same method implemented above
//      using POST method
//      ----------------------------------

        this.model = model;

        System.out.println();
        System.out.printf("editPromptES: %s", editPromptES);
        System.out.println();
        System.out.printf("editPromptEN: %s", editPromptEN);
        System.out.println();
        System.out.printf("editPromptPT: %s", editPromptPT);
        System.out.println();
        System.out.printf("editPromptDT: %s", editPromptDT);
        System.out.println();
        System.out.printf("editPromptFR: %s", editPromptFR);
        System.out.println();

        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss");
        this.model.addAttribute("currentDate", LocalDateTime.now().format(formatter));

        String url = "/public/home";
        String prompt;

        //prompt = "Hello";
        prompt = editPromptEN;
        if (!prompt.isEmpty() && !this.checkExist(prompt)) {
            this.addPrompt(this.model, prompt);
        }
//
        //prompt = "Buenos días, ¿Cómo estás?.";
        prompt = editPromptES;
        if (!prompt.isEmpty() && !this.checkExist(prompt)) {
            this.addPrompt(this.model, prompt);
        }

        //prompt = "Guten Tag, mein Freund";
        prompt = editPromptDT;
        if (!prompt.isEmpty() && !this.checkExist(prompt)) {
            this.addPrompt(this.model, prompt);
        }

        //prompt = "Vous ...?";
        prompt = editPromptFR;
        if (!prompt.isEmpty() && !this.checkExist(prompt)) {
            this.addPrompt(this.model, prompt);
        }

        //prompt = "Como vai?";
        prompt = editPromptPT;
        if (!prompt.isEmpty() && !this.checkExist(prompt)) {
            this.addPrompt(this.model, prompt);
        }

        this.updateData();
        this.loadPrompts(this.model);

//      ------------------------------------
//      code already tested and commented (out of execution)
//      ------------------------------------
        // this is for testing purpose. -testing POST method
//        Map<String, String> d = new HashMap<>();
//
//        d.put("taskId", UUID.randomUUID().toString());
//
//        data.add(d);
//
//        data.getFirst().put("prompt", this.decodeEncoding(prompt));
//        data.getFirst().put("response", this.decodeEncoding(""));
//        data.getFirst().put("error", this.decodeEncoding(""));
//        data.getFirst().put("status", this.translate(TaskStatus.PENDING));
//
//        model.addAttribute("data", data);

        // end testing

        return url;
    }

//    private String handleUrlRequest(Model model){
//        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss");
//        model.addAttribute("currentDate", LocalDateTime.now().format(formatter));
//
//        String url = "/public/home";
//        String prompt;
//
//        prompt = "Hello";
//        if (!this.checkExist(prompt)) {
//            this.addPrompt(model, prompt);
//        }
//
//        prompt = "Buenos días, ¿Cómo estás?.";
//        if (!this.checkExist(prompt)) {
//            this.addPrompt(model, prompt);
//        }
//
//        prompt = "Guten Tag, mein Freund";
//        if (!this.checkExist(prompt)) {
//            this.addPrompt(model, prompt);
//        }
//
//        prompt = "Tell me who is it?";
//        if (!this.checkExist(prompt)) {
//            this.addPrompt(model, prompt);
//        }
//
//        prompt = "Como vai?";
//        if (!this.checkExist(prompt)) {
//            this.addPrompt(model, prompt);
//        }
//
//        this.updateData();
//        this.loadPrompts(model);
//
//        return url;
//    }

    private void addPrompt(Model model, String prompt) {
        if (prompt.isEmpty()) { return; }

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
                    String strPrompt = promptStore.readPrompt(taskId).getPrompt();
                    String strResponse = promptStore.readPrompt(taskId).getResponse()
                                    .isEmpty() ?
                                        "":
                                        String.join(
                                                " ",
                                                promptStore.readPrompt(taskId).getResponse()
                                        ).replace(
                                                "\\n", "<br/>"
                                        ).replace(
                                                "\\", ""
                                        );
                    String strError = promptStore.readPrompt(taskId).getError()
                            .isEmpty() ?
                            "":
                            String.join(
                                    " ",
                                    promptStore.readPrompt(taskId).getError()
                            ).replace(
                                    "\\n", "<br/>"
                            ).replace(
                                    "\\", ""
                            );

                    data.get(index).put("prompt", this.decodeEncoding(strPrompt));
                    data.get(index).put("response", this.decodeEncoding(strResponse));
                    data.get(index).put("error", this.decodeEncoding(strError));
                    data.get(index).put("status", this.translate(statusStore.readStatus(taskId)));
                }

                index++;
            }
        }
    }

    private String decodeEncoding(String text) {
        ByteBuffer buffer = StandardCharsets.ISO_8859_1.encode(text);
        return StandardCharsets.ISO_8859_1.decode(buffer).toString();
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
            Thread.currentThread().interrupt();

            List<String> message = new ArrayList<>(){};
            message.add(e.getMessage());

            promptStore.getPrompt(taskId).setError(message);

            model.addAttribute("taskId", taskId);
            model.addAttribute("status", statusStore.readStatus(taskId));

            this.updateData();

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
                List<String> message = new ArrayList<>(){};
                message.add(e.getMessage());

                //promptStore.getPrompt(taskId).setError(message);

                //List<String> message = new ArrayList<>(){};
                //message.add(e.getMessage());

                promptStore.getPrompt(taskId).setError(message);
                this.updateData();

                model.addAttribute("taskId", taskId);
                model.addAttribute("status", statusStore.readStatus(taskId));

                model.addAttribute("prompt", promptStore.readPrompt(taskId).getPrompt());
                model.addAttribute("response", "");

                model.addAttribute("error", String.join(" ", promptStore.readPrompt(taskId).getError()).replace("\n",
                        "<br/>"));

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