# Experiment 13: Fuzzy Logic Controller (FLC) for Mobile Robot Obstacle Avoidance and Speed Control

## Department of Artificial Intelligence & Data Science

### Data Mining & Warehousing Laboratory

**Course Instructor:** Dr. Rajesh Kumar

---

## Overview

This experiment demonstrates the design and implementation of a **Mamdani Fuzzy Logic Controller (FLC)** for intelligent mobile robot navigation. The controller enables a robot to perform obstacle avoidance and speed control by using fuzzy logic rules based on sensor inputs.

Unlike conventional control systems, fuzzy logic handles uncertainty and imprecise information effectively, making it highly suitable for robotics and autonomous systems.

---

## Objective

The objectives of this experiment are:

* Understand the fundamentals of Fuzzy Logic Control.
* Design a Mamdani Fuzzy Inference System (FIS).
* Create linguistic variables and membership functions.
* Develop fuzzy rule bases for obstacle avoidance.
* Generate speed and steering control actions.
* Analyze robot behavior under different environmental conditions.

---

## Theory

### What is Fuzzy Logic?

Fuzzy Logic is an intelligent decision-making approach that allows reasoning with uncertain or approximate information.

Unlike classical logic, where values are strictly:

* 0 (False)
* 1 (True)

Fuzzy Logic permits intermediate truth values between 0 and 1.

### Applications of Fuzzy Logic

* Robotics
* Autonomous Vehicles
* Industrial Automation
* Intelligent Control Systems
* Smart Navigation Systems

---

## Fuzzy Logic Controller Structure

The Fuzzy Logic Controller consists of four major components:

1. Fuzzification
2. Rule Base
3. Inference Engine
4. Defuzzification

---

## Problem Statement

A mobile robot operates in an indoor environment and must:

* Detect nearby obstacles
* Avoid collisions
* Maintain smooth movement
* Adjust steering direction intelligently

### Sensor Inputs

* Distance Sensor (Ultrasonic / LiDAR)
* Angle Sensor (Camera / Vision / LiDAR)

### Controller Outputs

* Robot Speed
* Steering Direction

---

## Input Variables

### Distance

Range:

0 – 100 cm

Membership Functions:

| Membership Function | Type       | Parameters   |
| ------------------- | ---------- | ------------ |
| Near                | Triangular | [0 0 40]     |
| Medium              | Triangular | [20 50 80]   |
| Far                 | Triangular | [60 100 100] |

---

### Angle

Range:

-90° to +90°

Membership Functions:

| Membership Function | Type       | Parameters  |
| ------------------- | ---------- | ----------- |
| Left                | Triangular | [-90 -90 0] |
| Center              | Triangular | [-30 0 30]  |
| Right               | Triangular | [0 90 90]   |

---

## Output Variables

### Speed

Range:

0 – 1

Membership Functions:

| Membership Function | Type       | Parameters    |
| ------------------- | ---------- | ------------- |
| Slow                | Triangular | [0 0 0.4]     |
| Medium              | Triangular | [0.3 0.5 0.7] |
| Fast                | Triangular | [0.6 1 1]     |

---

### Steering Direction

Range:

-1 to +1

| Membership Function | Type       | Parameters   |
| ------------------- | ---------- | ------------ |
| Left                | Triangular | [-1 -1 0]    |
| Straight            | Triangular | [-0.2 0 0.2] |
| Right               | Triangular | [0 1 1]      |

---

## Fuzzy Rule Base

### Safety Rules

1. IF Distance is Near AND Angle is Center THEN Speed is Slow AND Steering is Right
2. IF Distance is Near AND Angle is Left THEN Speed is Slow AND Steering is Right
3. IF Distance is Near AND Angle is Right THEN Speed is Slow AND Steering is Left

### Medium Distance Rules

4. IF Distance is Medium AND Angle is Left THEN Speed is Medium AND Steering is Right
5. IF Distance is Medium AND Angle is Right THEN Speed is Medium AND Steering is Left
6. IF Distance is Medium AND Angle is Center THEN Speed is Slow AND Steering is Right

### Free Path Rules

7. IF Distance is Far AND Angle is Center THEN Speed is Fast AND Steering is Straight
8. IF Distance is Far AND Angle is Left THEN Speed is Fast AND Steering is Right
9. IF Distance is Far AND Angle is Right THEN Speed is Fast AND Steering is Left

---

## Inference System Parameters

| Parameter       | Method   |
| --------------- | -------- |
| AND             | Min      |
| OR              | Max      |
| Implication     | Min      |
| Aggregation     | Max      |
| Defuzzification | Centroid |

### Why Centroid Defuzzification?

* Produces smooth control outputs
* Reduces sudden steering changes
* Suitable for robotics applications
* Provides stable navigation

---

## Algorithm

1. Start MATLAB Fuzzy Logic Toolbox.
2. Create Mamdani FIS.
3. Define input variables.
4. Define output variables.
5. Add membership functions.
6. Create fuzzy rule base.
7. Configure inference methods.
8. Evaluate the FIS using test inputs.
9. Visualize outputs using Rule Viewer and Surface Viewer.
10. Analyze robot behavior.

---

## Observations

* Robot slows down near obstacles.
* Steering direction changes smoothly.
* Fuzzy rules mimic human driving behavior.
* Overlapping membership functions prevent abrupt actions.
* Mamdani FIS provides stable navigation.

---

## Result

The Mamdani Fuzzy Logic Controller was successfully designed and implemented for obstacle avoidance and speed control.

The controller effectively adjusted robot speed and steering direction according to obstacle distance and angle, resulting in safe and intelligent navigation.

---

## Applications

* Autonomous Mobile Robots
* Self-Driving Vehicles
* Automated Guided Vehicles (AGVs)
* Intelligent Wheelchairs
* Drone Navigation
* Warehouse Automation
* Robotics and AI Systems

---

## Advantages

* Handles uncertainty effectively
* Human-readable rule base
* Smooth control actions
* No exact mathematical model required
* Easy implementation for real-world systems

---

## Limitations

* Rule base grows with system complexity
* Requires expert knowledge for rule design
* Performance depends on membership function selection

---

## Conclusion

This experiment successfully demonstrated the implementation of a Mamdani Fuzzy Logic Controller for intelligent robot navigation. The controller used linguistic rules and fuzzy reasoning to generate smooth speed and steering outputs in uncertain environments. Fuzzy Logic proved to be a powerful AI technique for robotics, automation, and autonomous navigation applications.

---

## Software Requirements

* MATLAB
* Fuzzy Logic Toolbox

---

## Author

**Dr. Rajesh Kumar**
Department of Artificial Intelligence & Data Science
Data Mining & Warehousing Laboratory

