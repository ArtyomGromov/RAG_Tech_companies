def evaluate_retrieval(pipeline, queries, expected_pages, top_k=1, silent=True):
    """
    Выполняет оценку retrieval-пайплайна.

    Для каждого запроса:
      - Выполняется поиск с помощью pipeline.search_query().
      - Выводится найденный чанк (начало текста) и номер страницы.
      - Сравнивается найденный номер страницы с ожидаемым.

    В конце функция выводит общую точность (accuracy) по выбранным страницам.

    :param pipeline: Объект пайплайна (например, DensePDFRAGPipeline), в котором определён метод search_query().
    :param queries: Список строк с запросами.
    :param expected_pages: Список ожидаемых номеров страниц для каждого запроса.
    :param top_k: Количество возвращаемых результатов (по умолчанию 1).
    :param silent: Если False, выводит подробную информацию.
    :return: Точность (accuracy) в виде числа от 0 до 1.
    """
    correct = 0
    total = len(queries)

    for idx, query in enumerate(queries):
        result = pipeline.search_query(query, top_k=top_k)[0]
        expected_page = expected_pages[idx]
        if not silent:
            print(f"\nВопрос {idx+1}: {query}")
            print("Ожидаемая страница:", expected_page)
            print("Найденный чанк взят со страницы:", result["page"])
            print("Начало текста чанка:")
            print(result["chunk"][:200] + "...")
        if result["page"] == expected_page:
            if not silent:
                print("✅ Номер страницы совпадает с ожидаемым!")
            correct += 1
        else:
            if not silent:
                print("❌ Номер страницы НЕ совпадает с ожидаемым!")

    accuracy = correct / total if total > 0 else 0
    print(f"\nОбщая точность: {accuracy * 100:.2f}% ({correct}/{total})")
    return accuracy