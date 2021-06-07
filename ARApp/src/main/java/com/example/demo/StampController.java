package com.example.demo;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.servlet.ModelAndView;

@Controller
public class StampController {

	@RequestMapping(value = "/stamp", method = RequestMethod.GET)
	public ModelAndView stamp(ModelAndView mav) {
		mav.setViewName("stamp");
		return mav;

	}

	@RequestMapping(value = "/stamp/arcamera", method = RequestMethod.GET)
	public ModelAndView arcamera(ModelAndView mav) {
		mav.addObject("user", "user");
		mav.setViewName("arcamera2");
		return mav;

	}

	@RequestMapping(value = "/stamp/stampget/{num}", method = RequestMethod.GET)
	public ModelAndView stampget(@PathVariable String num, ModelAndView mav) {
		mav.addObject("num", num);
		mav.setViewName("stampget");
		return mav;

	}

}
