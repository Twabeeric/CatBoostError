# CatBoost Errors

During the IEEE CIS Fraud competition, I spent alot of time debugging my Catboot model and I wanted to save anyone else the trouble.

## Common Errors
### ValueError: could not convert string to float:
Separate your features into a list for numeric features and categorical features. Make sure no categorical features are listed as numeric features as catboost assumes anything not input as categorical is numeric.
Check your indices also, they are zero based indices and a cat_features indice should align with the correct indice for your feature names.


### Raise CatboostError("Invalid cat_features type={}: must be list() or np.ndarray()."
The dtype for your cat_features input must be a list or array

### CatboostError: Incorrect CD file. Invalid line number #10: catboost/libs/column_description/cd_parser.cpp:60: Invalid column index: index = 10, columnsCount = 10
Cat_features and feature_names are input as zero based indices usually with the label as the zero index (label=0)
For feature_names:

feature_names = dict()
for column, name in enumerate(train_df):
    if column == 0:
        continue
    feature_names1[column-1] = name     ###Notice (column-1)

For cat_features:
for i in catlist:
    cat_featuresls.append(train_df.columns.get_loc(i)+1)   ###Your cat_features index should begin at 1 

For feature selection:

features_to_evaluate = [0, 1, 2]  ###Notice that the first feature is index 0, the model skips the target label listed as 0.

### CatboostError: Invalid cat_features[55] = 451 value: must be < 446..
Check the length of your indices, length of indices should be one less than total number of features.

 ### CatBoostError: Error in dsv data. Line 1: Column 54 (type Num, value = "F"): catboost/libs/data_new/cb_dsv_loader.cpp:146: Factor 53 cannot be parsed as float. Try correcting column description file
Misaligned indices for cat_features leading to string cat_features being input as numeric features

### CatBoostError Max target greater than 1: 3.57754e+06
Error with choosing the loss function, probabilities should not be greater than 1.


Right now, we’re in your first GitHub **repository**. A repository is like a folder or storage space for your project. Your project's repository contains all its files such as code, documentation, images, and more. It also tracks every change that you—or your collaborators—make to each file, so you can always go back to previous versions of your project if you make any mistakes.

This repository contains three important files: The HTML code for your first website on GitHub, the CSS stylesheet that decorates your website with colors and fonts, and the **README** file. It also contains an image folder, with one image file.

## Describe your project

You are currently viewing your project's **README** file. **_README_** files are like cover pages or elevator pitches for your project. They are written in plain text or [Markdown language](https://guides.github.com/features/mastering-markdown/), and usually include a paragraph describing the project, directions on how to use it, who authored it, and more.

[Learn more about READMEs](https://help.github.com/en/articles/about-readmes)

## Your first website

**GitHub Pages** is a free and easy way to create a website using the code that lives in your GitHub repositories. You can use GitHub Pages to build a portfolio of your work, create a personal website, or share a fun project that you coded with the world. GitHub Pages is automatically enabled in this repository, but when you create new repositories in the future, the steps to launch a GitHub Pages website will be slightly different.

[Learn more about GitHub Pages](https://pages.github.com/)

## Rename this repository to publish your site

We've already set-up a GitHub Pages website for you, based on your personal username. This repository is called `hello-world`, but you'll rename it to: `username.github.io`, to match your website's URL address. If the first part of the repository doesn’t exactly match your username, it won’t work, so make sure to get it right.

Let's get started! To update this repository’s name, click the `Settings` tab on this page. This will take you to your repository’s settings page. 

![repo-settings-image](https://user-images.githubusercontent.com/18093541/63130482-99e6ad80-bf88-11e9-99a1-d3cf1660b47e.png)

Under the **Repository Name** heading, type: `username.github.io`, where username is your username on GitHub. Then click **Rename**—and that’s it. When you’re done, click your repository name or browser’s back button to return to this page.

<img width="1039" alt="rename_screenshot" src="https://user-images.githubusercontent.com/18093541/63129466-956cc580-bf85-11e9-92d8-b028dd483fa5.png">

Once you click **Rename**, your website will automatically be published at: https://your-username.github.io/. The HTML file—called `index.html`—is rendered as the home page and you'll be making changes to this file in the next step.

Congratulations! You just launched your first GitHub Pages website. It's now live to share with the entire world

## Making your first edit

When you make any change to any file in your project, you’re making a **commit**. If you fix a typo, update a filename, or edit your code, you can add it to GitHub as a commit. Your commits represent your project’s entire history—and they’re all saved in your project’s repository.

With each commit, you have the opportunity to write a **commit message**, a short, meaningful comment describing the change you’re making to a file. So you always know exactly what changed, no matter when you return to a commit.

## Practice: Customize your first GitHub website by writing HTML code

Want to edit the site you just published? Let’s practice commits by introducing yourself in your `index.html` file. Don’t worry about getting it right the first time—you can always build on your introduction later.

Let’s start with this template:

```
<p>Hello World! I’m [username]. This is my website!</p>
```

To add your introduction, copy our template and click the edit pencil icon at the top right hand corner of the `index.html` file.

<img width="997" alt="edit-this-file" src="https://user-images.githubusercontent.com/18093541/63131820-0794d880-bf8d-11e9-8b3d-c096355e9389.png">


Delete this placeholder line:

```
<p>Welcome to your first GitHub Pages website!</p>
```

Then, paste the template to line 15 and fill in the blanks.

<img width="1032" alt="edit-githuboctocat-index" src="https://user-images.githubusercontent.com/18093541/63132339-c3a2d300-bf8e-11e9-8222-59c2702f6c42.png">


When you’re done, scroll down to the `Commit changes` section near the bottom of the edit page. Add a short message explaining your change, like "Add my introduction", then click `Commit changes`.


<img width="1030" alt="add-my-username" src="https://user-images.githubusercontent.com/18093541/63131801-efbd5480-bf8c-11e9-9806-89273f027d16.png">

Once you click `Commit changes`, your changes will automatically be published on your GitHub Pages website. Refresh the page to see your new changes live in action.

:tada: You just made your first commit! :tada:

## Extra Credit: Keep on building!

Change the placeholder Octocat gif on your GitHub Pages website by [creating your own personal Octocat emoji](https://myoctocat.com/build-your-octocat/) or [choose a different Octocat gif from our logo library here](https://octodex.github.com/). Add that image to line 12 of your `index.html` file, in place of the `<img src=` link.

Want to add even more code and fun styles to your GitHub Pages website? [Follow these instructions](https://github.com/github/personal-website) to build a fully-fledged static website.

![octocat](./images/create-octocat.png)

## Everything you need to know about GitHub

Getting started is the hardest part. If there’s anything you’d like to know as you get started with GitHub, try searching [GitHub Help](https://help.github.com). Our documentation has tutorials on everything from changing your repository settings to configuring GitHub from your command line.
