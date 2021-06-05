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
		mav.setViewName("arcamera");
		return mav;

	}

	@RequestMapping(value = "/stamp/stampget/{num}", method = RequestMethod.GET)
	public ModelAndView stampget(@PathVariable int num, ModelAndView mav) {
		mav.setViewName("stampget");
		mav.addObject("msg", num);
		return mav;

	}

}
