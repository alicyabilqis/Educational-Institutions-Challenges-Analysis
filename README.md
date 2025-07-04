# A Comprehensive Approach to Identifying and Reducing Student Dropout Rates: Performance Monitoring and Early Intervention Strategies in Educational Institutions
Student dropout is a persistent challenge that affects educational institutions globally, regardless of size or ranking. While some students leave for personal or financial reasons, others gradually disengage due to academic struggles, a lack of motivation, or limited access to academic support services. Regardless of the cause, the implications of student dropouts are significant.

From an institutional perspective, a high dropout rate can impact accreditation standings, public reputation, financial sustainability (especially where funding is linked to student retention), and internal morale among faculty and staff. For students, dropping out can result in long-term economic disadvantages, reduced employment opportunities, and diminished self-confidence.

With the increasing availability of digital student data—ranging from grades and attendance to demographic background and engagement in learning platforms—educational institutions now have the opportunity to implement data-driven solutions that identify students at risk. These insights can be used to design tailored intervention programs that increase student retention and support academic success.

This project seeks to explore how machine learning and data visualization tools can be applied to proactively monitor student performance and flag those who may be at risk of dropping out. By moving from reactive to proactive support strategies, institutions can create more equitable and effective learning environments.

### Problem Statement
Many educational institutions still rely on manual methods or outdated reporting systems to monitor student performance and engagement. This often results in delayed interventions, where students who are struggling are only noticed once they have already disengaged or failed key assessments. The absence of early warning systems leads to inefficient resource allocation and missed opportunities for academic support.

The challenge lies in developing a system that can accurately identify patterns in student data predictive of dropouts, provide timely alerts to faculty and academic advisors before students reach a critical point, and display insights through an intuitive dashboard that supports informed decision-making for interventions.

Therefore, this project aims to answer the following key questions:
* What are the most important academic and demographic features that predict student dropout?
* How can predictive modeling be used to classify students based on their dropout risk?
* How can the results be visualized in a user-friendly dashboard to help administrators monitor performance in real time?
* What actionable strategies can be recommended based on the model’s insights?

### Project Scope
This project will deliver a data-driven solution for monitoring student performance and predicting dropout risks. The scope includes:
1. **Data Understanding and Exploration**  
   Analyze the provided dataset to understand student attributes, academic performance, and dropout patterns.
   
2. **Data Preprocessing**  
   Clean the dataset, handle missing values, encode categorical features, and prepare it for modeling.
   
3. **Exploratory Data Analysis (EDA)**  
   Uncover relationships between variables and identify trends associated with dropouts.
   
4. **Predictive Modeling**  
   Build and evaluate machine learning models to predict dropout likelihood.
   
5. **Dashboard Development**  
   Create an interactive dashboard to monitor student performance and visualize dropout risks.
   
6. **Insights and Recommendations**  
   Provide actionable insights and recommend strategies to support at-risk students early in the academic cycle.
#### Preparation
The data used for this analysis was sourced from [students performance data](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance 'Dicoding GitHub - students_performance')

Setup environment :
1. Clone the Repository
```sh
git clone https://github.com/alicyabilqis/HR-Department-Problem-Analysis.git
```
2. Create a Virtual Environment
```sh
virtualenv venv
```
3. Activate the Environment
```sh
venv\Scripts\activate
```
4. Install Dependencies
```sh
pip install -r requirements.txt
```
5. Menjalankan app
```sh
streamlit run app.py
```
## Business Dashboard

#### Visual Summary of Findings
![Business Dashboard](https://github.com/alicyabilqis/Educational-Institutions-Challenges-Analysis/raw/main/Alicya-dashboard.jpg)
#### Visual Interpretation
This report analyzes the challenges faced by educational institutions based on data from 4,424 students. Among these, 2,209 students successfully graduated (49.9%), 1,421 students dropped out (32.1%), and 794 students are still active or currently pursuing their studies (17.9%). The dropout rate, which accounts for more than one-third of the total student population, highlights a serious issue that requires systematic intervention.

When examining age at enrollment, the 17–20 age group shows the highest number of dropouts. This suggests that younger students tend to be more vulnerable to academic failure or difficulties adjusting to campus life. A similar trend is observed among students aged 21–23, who also exhibit a relatively high dropout rate.

In terms of tuition payment, students who paid their fees in full had significantly higher graduation rates compared to those who did not. A majority of students who did not pay their tuition ended up dropping out. This underscores the critical role that financial stability plays in academic success.

Initial academic ability, measured by admission grade, also has a significant impact on graduation outcomes. Students with average or below-average admission scores were more likely to drop out, while those with higher grades tended to graduate. This indicates the need for additional academic support for students with weaker academic backgrounds.

From a program perspective, vocational fields such as Nursing and Social Services recorded the highest number of graduates. However, certain programs—such as Voluntary Nursing, Management, and Journalism & Communication—had higher dropout numbers. This may be due to challenging curricula, unmet job expectations, or lack of adequate academic support.

An analysis of academic performance in the first and second semesters reveals that students who graduated tended to complete more courses with good grades and evaluations. In contrast, dropout students had fewer approved courses and lower academic performance from the start. This highlights the importance of early academic intervention beginning in the first semester.

Overall, the key factors contributing to student dropout are low admission grades, financial difficulties, poor academic performance from the outset, and younger age—possibly reflecting a lack of preparedness for the demands of higher education.
#### Documentation
The Business Dashboard is accessible via the following link:

[View the Dashboard](https://lookerstudio.google.com/reporting/8bb1f9df-c52d-468d-8af9-43b153dceb96)

To interact with and test the predictive model, please refer to the following link:

[View the Prediction Model](https://educational-institutions-challenges-analysis-abr8mnmkwfjv3r3fy.streamlit.app/)
## Conclusion
### Business Dashboard
This report analyzes data from 4,424 students to identify key challenges faced by educational institutions, revealing a concerning dropout rate of 32.1%. Younger students (ages 17–23) are more prone to dropping out, likely due to academic unpreparedness and difficulty adjusting to college life. Financial factors also play a critical role, as students who paid their tuition in full had significantly higher graduation rates than those who did not. Academic background, particularly low admission grades, strongly correlates with dropout risk, as does poor performance in the first two semesters. Certain programs—such as Voluntary Nursing, Management, and Journalism—show higher dropout rates, potentially due to demanding curricula or lack of support. Overall, dropout is driven by a combination of financial hardship, weak academic foundations, and early academic struggles.
### Predictive model
The model has identified a combination of academic performance metrics, administrative status, and demographic factors as key predictors of student dropout. The model shows high sensitivity to students' academic progression—particularly in the first and second semesters—highlighting these early stages as critical intervention points. Financial and enrollment-related variables, such as tuition payment status and age at enrollment, also contribute to dropout risk but to a lesser extent.

The model's top predictors are strongly academic in nature, especially:
* The number of courses passed and grades in both semesters,
* The evaluation load in each term, and
* The student’s academic background prior to enrollment.

This suggests that student success in the first year is the most important period for retention and support efforts.
### Recommended Actions
To reduce dropout rates effectively, educational institutions should implement a combination of academic, financial, and structural interventions targeting the top predictors:

1. Academic Monitoring and Intervention
   * Create real-time dashboards to flag students with low pass rates or declining grades.
   * Offer personalized academic support including mentoring, tutoring, and academic coaching—especially for students underperforming in the first and second semesters.
   * Integrate remedial or recovery programs (e.g., modular repeats or summer classes) for students falling behind.

2. Curriculum and Assessment Management
   * Evaluate and optimize the number and type of assessments in early semesters to avoid overburdening students.
   * Encourage continuous and formative assessments rather than high-stakes exams.

3. Financial Support Mechanisms
   * Set up early alerts for students falling behind on tuition payments and connect them with financial aid advisors.
   * Provide flexible payment plans and establish emergency grant programs to alleviate financial stress.

4. Support for Non-Traditional Students
   * Offer flexible course formats (evening, online, or hybrid) for older students or those with non-academic obligations.
   * Facilitate peer communities based on age groups or personal responsibilities to strengthen belonging and motivation.

5. Pre-University Preparation
   * Use admission grades as a baseline to assign at-risk students to bridging or academic success programs.
   * Implement diagnostic assessments at the beginning of the academic year to tailor support services to student needs.
