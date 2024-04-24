package com.bomdemo.kimiapi.controller;

import com.bomdemo.kimiapi.model.ChatContent;
import com.bomdemo.kimiapi.model.ResponseChoice;
import com.bomdemo.kimiapi.service.ChatService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import javax.annotation.Resource;
import java.util.List;

@RestController
@RequestMapping("/aichat/v1/base")
@Slf4j
public class MessageConnection {

    @Resource
    private ChatService chatService;

    @PostMapping({"/enter"})
    public ResponseChoice list(@RequestBody ChatContent message) throws Exception {
        ResponseChoice res = chatService.sentMessage(message);
        return res;
    }

}
