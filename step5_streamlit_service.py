import streamlit as st
from openai import OpenAI

from steps.step4_retrieval import retrieve_similar

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    base_url="https://api.proxyapi.ru/openai/v1",
)


def get_course_recommendation(user_query: str):
    """
    Получает рекомендации по курсам из нашей базы знаний
    """
    try:
        similar_courses = retrieve_similar(user_query, top_k=3)

        context = "\n\n".join(
            [f"Курс {url}:\n{summary}" for url, summary in similar_courses]
        )
        with open("prompts/user_prompt_template.txt", "r") as f:
            prompt = f.read()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt.format(user_query=user_query, context=context),
                },
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Произошла ошибка: {str(e)}"


def main():
    st.title("🎓 Ассистент по выбору курсов Karpov.Courses")

    st.write(
        """
    ### Добро пожаловать!
    Опишите ваши интересы и цели в обучении, и я помогу подобрать подходящие курсы 
    из каталога Karpov.Courses.
    """
    )

    user_input = st.text_area(
        "Опишите ваши интересы и цели:",
        height=150,
        placeholder="Например: Я хочу научиться анализировать данные и работать с Python...",
    )

    if st.button("Подобрать курсы"):
        if user_input:
            with st.spinner("Анализирую подходящие курсы..."):
                recommendation = get_course_recommendation(user_input)
                st.write("### Рекомендованные курсы:")
                st.write(recommendation)
        else:
            st.warning("Пожалуйста, опишите ваши интересы и цели обучения.")

    with st.sidebar:
        st.header("Как получить лучшие рекомендации")
        st.write(
            """
        1. Укажите конкретные области, которые вас интересуют
        2. Опишите свой текущий уровень знаний
        3. Расскажите о желаемых результатах обучения
        """
        )


if __name__ == "__main__":
    main()
