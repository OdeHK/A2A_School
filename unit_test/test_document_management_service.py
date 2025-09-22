from services.document_processing.document_management_service import DocumentManagementService


def test_get_document_id_dict():
    document_management = DocumentManagementService()
    document_management.load_session(session_id="session_20250918_203241_11f77b42")
    document_id_dict = document_management.get_document_id_dict()
    assert isinstance(document_id_dict, dict)
    assert all(isinstance(key, str) and isinstance(value, str) for key, value in document_id_dict.items())

    print(f"Document ID to File Name mapping in the session: {document_id_dict}")
    print(document_id_dict)

if __name__ == "__main__":
    test_get_document_id_dict()
    print("All tests passed.")