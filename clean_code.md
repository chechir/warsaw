# Quick review of clean code book:

chapter1: clean code
    Bad code
    The total cost of owning a mess
    The great redesign in the Sky
    The art of clean code
    Prequel and Principles

chapter2: Meaninful names

chapter3: Functions

chapter4: Comments

chapter5: Formatting

chapter6: Objects and Data Structures
        Levels of abstractions


# Workshop ##

### use case:
Particularly apply to kaggle:
Bad code in Kaggle:
    when I was doing my third competition, my strategy was to use bag many xgboost predictions.
    So, I fitted models and predict on the test set overnight, and in the morning I took
    those predictions and average them with older predictions.
    Then the next day I did the same, copying the code into a new file and changing the hypermarameters
    and features.  I kept this going for several weeks and
    ended up with a huge pile of code and predictions that got me to the 6th position.
    Usually I had to deal with things like: Introducing bugs in my feature functions, deleting
    inadvertly some portion of useful code, copying and pasting the wrong code, etc.
    The code was a long sheet and I struggled to find my way, hoping for some hint of what was
    going on.

    Then, I got invited to participate in a team with 4 more people. Because I wasn't sure I could
    reproduce my predictions and the code was so duplicated, I had to start over with a much simpler
    model. Luckily, I could use my best features and we didn't lose too much score.
    But I wasted a lot of time trying to understand and tie up all the different files with duplicated
    code.  Time in kaggle is critical. I think if I had use cleaner code from the beggining probably
    we could have done better, or at least had more time to try more ideas

    In kaggle, we may want to go fast and go quickly to the top of the leaderboard, so you
    are usually in a rush. I like to think on Kaggle competitions now not like it were only
    one competition, rather, I like to think on building a machine that can do well in Kaggle
    always. To have a pipeline that alows you to reuse features, models, data cleaning processes
    from old competitions, and not to have to waste time every time with the same things

    This process also helps you to work on meaninful things, and not just change some parameters
    to get an extra 0.0001% of improvement that will make you go higer on the LB

Clean code:
    like painting: Everbody recognises it but not everyone can make it
    Elegant, pleasent
    Details matter. Like in a painting, the picture is made of details
        error handling: is a way to gloss over details
    it follows the principle of less surprise

DRY
Dont repeat yourself
    you create two versions of something. When you need to change or improve your functionality
    you need to change more than one place and that can introduce bugs in your code


Meaninful names

Functions:
    *Do one thing*
    short
    use one level of abstraction
    use few parameters


#####################################################

TDD

Why 'guided' test?

# Chapter 1
acceptance test
unit test

end to end: I dont use it: Deployment test from the start maybe?
  without calling directly its internal code
   (I assume that mocks everything? tests the user interface concept)


whole cycle: 
golden rule: never write a new functionality without a failing test

Automated build
  check last version
  compile test code
  integrate and package the system
  exercise the system through its external access points

Levels
  - acceptance
  - Integration
  - Unit


External and internal quality
 why a test may be difficult to write. 
  Listen to tests


## Chapter 4
Deploying and testing from the start: forces to understand how the system fits into the real world. Flushes out the 'unknown unknown'
  Problem of testing the first feature (first-feature paradox):
    a) Work out how to build, deploy and test a 'walking skeleton'
    b) Then test write acceptance tests for the first meaningful feature. 
' Walking skeleton' : implementation of the thinest possible slice of real funcitionality that we can automatically build, deploy, and test ent-to-end

design the walink skeleton: been able to draw it on the board in few minutes

Understanding of th purpuse of the system
Build sources of feedback: releases regularly  to a real production system
   Here: the automated deployment

### Expose uncertainty earlier:
  involves asking a lot of awkward questions
  Frontloads the stress of the project
 Limit to the white board level discussions

## Cahpter 5

### start each feature with an acceptance test

We write the acceptance test using only terminology from the application (not bd or web service). Independent of underlying technologies
Clarify what we want to achieve

Acceptance tests: Helps us to understand the user' perspective
Unit test: exercise objects, small clusters of objects, in isolation. They don't say anithing about whether they work together wiht the rest of the system

New acceptance tests





    unit tests
    regression tests
