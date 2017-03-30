---
layout: post
title: "Setting up Jekyll on Windows"
description: ""
category: 
tags: [Jekyll]
---


This blog is powered by [Jekyll-3.2.1](http://jekyllbootstrap.com/usage/jekyll-quick-start.html).  

I followed the instructions on [Jekyll QuickStart](http://jekyllbootstrap.com/usage/jekyll-quick-start.html) to set up Jekyll however they seemed incomplete.  I got the error message "*cannot load such file -- bundler LoadError*" when I executed the command "*jekyll serve*" on windows. 

After goolging, I managed to fix the problem by following the steps below: 

 * Install the **bundler** gem by executing "*gem install bundler*"
 * Download the **Ruby Development Kit** from this [page](http://rubyinstaller.org/downloads/) and extract the file to a DevKit folder 
 * In the DevKit folder, hold Shift then click the right mouse button, select "Open command window". On the command window,  run "*ruby dk.rb init*", then "*ruby dk.rb install*" to bind it to ruby installations in your path.
 * Install the **pygments** gem by executing "*gem install pygments.rb*" 
 * Install the **jekyll-sitemap** gem by executing "*gem install jekyll-sitemap*" 